import argparse
import os
from contextlib import contextmanager

import accelerate
import cv2
import einops
import numpy as np
import torch
from einops import rearrange
from loguru import logger
from omegaconf import OmegaConf
from torch._C import _distributed_c10d
from torch.nn.functional import adaptive_avg_pool2d
from tqdm.auto import tqdm

import models.vqgan
from libs.inception import InceptionV3
from models.uvit import UViT


_distributed_c10d.set_debug_level(_distributed_c10d.DebugLevel.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))


@contextmanager
def pushd(path):
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(BASE_DIR, path))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join(BASE_DIR, "configs", "maskgit.yaml"))
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--benchmark", type=int, default=0)
    parser.add_argument("--test_bsz", type=int, default=64)
    parser.add_argument("--gen_step", type=int, default=8)
    parser.add_argument("--cfg", action="store_true", default=False)
    parser.add_argument("--pre_full_iters", type=int, default=0)
    parser.add_argument("--num_cache_iters", type=int, default=0)
    parser.add_argument("--eval_n", type=int, default=50000)
    parser.add_argument("--samp_temp", type=float, default=0.7)
    parser.add_argument("--conf_temp", type=float, default=5.5)
    parser.add_argument(
        "--reference_image_path",
        type=str,
        default=os.path.join(REPO_ROOT, "assets", "fid_stats", "imagenet256_guided_diffusion.npz"),
    )
    parser.add_argument(
        "--inception_weight_path",
        type=str,
        default=os.path.join(REPO_ROOT, "assets", "pt_inception-2015-12-05-6726825d.pth"),
    )
    return parser.parse_args()


def calc_fid(m1, s1, m2, s2):
    mean_term = (m1 - m2).square().sum(dim=-1)
    trace_term = s1.trace() + s2.trace()
    cov_term = torch.linalg.eigvals(s1 @ s2).sqrt().real.sum(dim=-1)
    return (mean_term + trace_term - 2 * cov_term).item()


@logger.catch()
def decode(config, args):
    output_dir = resolve_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger.add(os.path.join(output_dir, "output.log"), level="INFO")

    if args.benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(args.seed, device_specific=True)

    if accelerator.is_main_process:
        logger.info("Setting seed: {}", args.seed)
    logger.info("Process {} using device: {}", accelerator.process_index, device)
    logger.info("Using mini-batch size {} per device", args.test_bsz)

    accelerator.wait_for_everyone()
    logger.info("Run on {} devices", accelerator.num_processes)

    if not os.path.isabs(config.nnet.pretrained_path):
        config.nnet.pretrained_path = os.path.join(BASE_DIR, config.nnet.pretrained_path)

    with np.load(args.reference_image_path) as f:
        m2 = torch.from_numpy(f["mu"][:]).to(device=device, dtype=torch.double)
        s2 = torch.from_numpy(f["sigma"][:]).to(device=device, dtype=torch.double)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx], weight_path=args.inception_weight_path).to(device)
    inception.eval()
    inception.requires_grad_(False)

    with pushd(BASE_DIR):
        autoencoder = models.vqgan.get_model()
    codebook_size = autoencoder.n_embed
    config.nnet.codebook_size = codebook_size
    autoencoder.to(device)

    @torch.cuda.amp.autocast(enabled=True)
    def decode_code(batch):
        return autoencoder.decode_code(batch)

    empty_ctx = torch.from_numpy(np.array([[1000]], dtype=np.longlong)).to(device)
    nnet_ema = UViT(**config.nnet)
    nnet_ema = accelerator.prepare(nnet_ema)
    ckpt = torch.load(config.nnet.pretrained_path, map_location="cpu")
    try:
        nnet_ema.module.load_state_dict(ckpt)
    except Exception:
        nnet_ema.load_state_dict(ckpt)
    nnet_ema.eval()

    def cfg_nnet(x, labels, scale, cond_past_kvs=None, uncond_past_kvs=None, use_cache=False, position_ids=None):
        cond, cond_past_kvs = nnet_ema(
            x, context=labels, past_kvs=cond_past_kvs, use_cache=use_cache, position_ids=position_ids
        )
        uncond, uncond_past_kvs = nnet_ema(
            x,
            context=einops.repeat(empty_ctx, "1 ... -> B ...", B=x.size(0)),
            past_kvs=uncond_past_kvs,
            use_cache=use_cache,
            position_ids=position_ids,
        )
        return cond + scale * (cond - uncond), cond_past_kvs, uncond_past_kvs

    def amortize(n_samples, batch_size):
        k = n_samples // batch_size
        r = n_samples % batch_size
        return k * [batch_size] if r == 0 else k * [batch_size] + [r]

    def get_data_generator():
        while True:
            yield torch.randint(0, 1000, (args.test_bsz, 1), device=device)

    def extract_features(samples):
        pred = inception(samples.float())[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        return pred.squeeze(3).squeeze(2)

    data_generator = get_data_generator()
    n_samples = args.eval_n
    batch_size = args.test_bsz * accelerator.num_processes

    mask_ind = codebook_size
    seq_len = 256
    hw = 16

    def add_gumbel_noise(t, temperature):
        return t + torch.tensor(temperature * np.random.gumbel(size=t.shape), device=device)

    ratio_schedule = 1 - np.linspace(0, 1, args.gen_step + 1)[:-1] ** 2.5
    ntoken_schedule = np.floor(ratio_schedule * seq_len).astype(int)
    ntoken_schedule = (ntoken_schedule[:-1] - ntoken_schedule[1:]).tolist()
    ntoken_schedule = ntoken_schedule + [seq_len - sum(ntoken_schedule)]
    for idx, ntoken in enumerate(ntoken_schedule):
        if ntoken == 0:
            ntoken_schedule[idx] = 1
            ntoken_schedule[-1] -= 1

    if args.num_cache_iters == 0:
        local_ntoken_schedules = [[i] for i in ntoken_schedule]
    else:
        local_ntoken_schedules = []
        cnt = 0
        while cnt < len(ntoken_schedule):
            local_ntoken_schedule = [ntoken_schedule[cnt]]
            cnt += 1
            if cnt > args.pre_full_iters:
                for _ in range(args.num_cache_iters):
                    if cnt < len(ntoken_schedule):
                        local_ntoken_schedule.append(ntoken_schedule[cnt])
                        cnt += 1
                    else:
                        break
            local_ntoken_schedules.append(local_ntoken_schedule)
        ntoken_schedule = [sum(i) for i in local_ntoken_schedules]
    assert sum(ntoken_schedule) == seq_len

    samp_temp_schedule = args.samp_temp + (1 - np.linspace(0, 1, args.gen_step) ** 0.5) * (1.0 - args.samp_temp)
    samp_temp_schedule = samp_temp_schedule.tolist()
    conf_temp_schedule = args.conf_temp * (1.0 - np.array([(i + 1) / len(ntoken_schedule) for i in range(len(ntoken_schedule))]))
    cfg_schedule = np.linspace(0.05, 0.1, args.gen_step).tolist()

    local_samp_temp_schedules = []
    local_cfg_schedules = []
    for local_schedule in local_ntoken_schedules:
        local_samp_temp_schedules.append(samp_temp_schedule[: len(local_schedule)])
        local_cfg_schedules.append(cfg_schedule[: len(local_schedule)])
        samp_temp_schedule = samp_temp_schedule[len(local_schedule) :]
        cfg_schedule = cfg_schedule[len(local_schedule) :]

    save_folder = os.path.join(
        output_dir,
        f"step{args.gen_step}_conf{args.conf_temp}_samp{args.samp_temp}_pre{args.pre_full_iters}_cache{args.num_cache_iters}",
    )
    os.makedirs(save_folder, exist_ok=True)

    gathered_features = []
    cnt = 0
    for _ in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc="sample"):
        with torch.no_grad():
            labels = next(data_generator)
            nnet = cfg_nnet if args.cfg else nnet_ema
            x = torch.full((labels.shape[0], seq_len), mask_ind, dtype=torch.long, device=device)
            for step, ntoken in enumerate(ntoken_schedule):
                batch_range = torch.arange(labels.shape[0], device=device).unsqueeze(-1).expand(-1, ntoken)
                is_mask = x.eq(mask_ind)
                if args.cfg:
                    local_cfg_schedule = local_cfg_schedules[step]
                    cfg_scale = local_cfg_schedule[0]
                    logits, full_cond_past_kvs, full_uncond_past_kvs = nnet(
                        x, labels, scale=cfg_scale, cond_past_kvs=None, uncond_past_kvs=None, use_cache=True
                    )
                else:
                    logits, full_past_kvs = nnet(x, context=labels, past_kvs=None, use_cache=True)

                local_ntoken_schedule = local_ntoken_schedules[step]
                local_gen_step = len(local_ntoken_schedule)
                local_samp_temp_schedule = local_samp_temp_schedules[step]

                x_sampled = torch.distributions.Categorical(logits=logits).sample()
                logits_sampled = torch.log_softmax(logits, dim=-1)
                logits_sampled = torch.gather(logits_sampled, dim=-1, index=x_sampled.unsqueeze(-1)).squeeze(-1)
                logits_sampled = torch.where(is_mask, logits_sampled, -np.inf).float()
                confidence = add_gumbel_noise(logits_sampled, conf_temp_schedule[step])
                _, sorted_idx = torch.sort(confidence, dim=-1, descending=True)
                unmask_idx = sorted_idx[:, :ntoken]
                cached_idx = sorted_idx[:, ntoken:]

                if len(local_ntoken_schedule) == 1:
                    x[batch_range, unmask_idx] = x_sampled[batch_range, unmask_idx]
                    continue

                local_x = torch.full((labels.shape[0], ntoken), mask_ind, dtype=torch.long, device=device)
                local_logits = logits[batch_range, unmask_idx]
                batch_range1 = torch.arange(labels.shape[0], device=device).unsqueeze(-1).expand(-1, seq_len - ntoken)
                if args.cfg:
                    cond_past_kvs = []
                    uncond_past_kvs = []
                    for cond_past_kv in full_cond_past_kvs:
                        cond_past_kvs.append(
                            (cond_past_kv[0][:, 1:][batch_range1, cached_idx], cond_past_kv[1][:, 1:][batch_range1, cached_idx])
                        )
                    for uncond_past_kv in full_uncond_past_kvs:
                        uncond_past_kvs.append(
                            (
                                uncond_past_kv[0][:, 1:][batch_range1, cached_idx],
                                uncond_past_kv[1][:, 1:][batch_range1, cached_idx],
                            )
                        )
                else:
                    past_kvs = []
                    for past_kv in full_past_kvs:
                        past_kvs.append((past_kv[0][:, 1:][batch_range1, cached_idx], past_kv[1][:, 1:][batch_range1, cached_idx]))

                s_idx = 0
                for local_step, local_ntoken in enumerate(local_ntoken_schedule):
                    e_idx = s_idx + local_ntoken
                    local_temp = local_samp_temp_schedule[local_step]
                    local_x[:, s_idx:e_idx] = torch.distributions.Categorical(logits=local_logits / local_temp).sample()[
                        :, s_idx:e_idx
                    ]
                    if local_step < local_gen_step - 1:
                        if args.cfg:
                            cfg_scale = local_cfg_schedule[local_step]
                            local_logits, _, _ = nnet(
                                local_x,
                                labels,
                                scale=cfg_scale,
                                cond_past_kvs=cond_past_kvs,
                                uncond_past_kvs=uncond_past_kvs,
                                use_cache=False,
                                position_ids=unmask_idx,
                            )
                        else:
                            local_logits, _ = nnet(
                                local_x, context=labels, past_kvs=past_kvs, use_cache=False, position_ids=unmask_idx
                            )
                    s_idx = e_idx
                assert local_x.eq(mask_ind).sum() == 0
                x[batch_range, unmask_idx] = local_x

        assert x.eq(mask_ind).sum() == 0
        z = rearrange(x, "b (i j) -> b i j", i=hw, j=hw)
        gen_images_batch = decode_code(z)
        gathered = accelerator.gather(extract_features(gen_images_batch.clamp_(0.0, 1.0)))
        if accelerator.is_main_process:
            gathered_features.append(gathered.cpu())

        gen_images_batch = gen_images_batch.detach().cpu()
        for b_id in range(gen_images_batch.size(0)):
            img_id = cnt * gen_images_batch.size(0) * accelerator.num_processes
            img_id += accelerator.process_index * gen_images_batch.size(0) + b_id
            if img_id >= n_samples:
                break
            gen_img = np.clip(gen_images_batch[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255)
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, f"{str(img_id).zfill(5)}.png"), gen_img)
        cnt += 1

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pred_tensor = torch.cat(gathered_features, dim=0)[:n_samples].to(device=device, dtype=torch.double)
        m1 = pred_tensor.mean(dim=0)
        pred_centered = pred_tensor - m1
        s1 = torch.mm(pred_centered.T, pred_centered) / (pred_tensor.size(0) - 1)
        fid = calc_fid(m1, s1, m2, s2)
        logger.info("FID{}={}", n_samples, fid)
        with open(os.path.join(output_dir, f"cache_fid{n_samples}.txt"), "a", encoding="utf-8") as f:
            f.write(
                f"FID{n_samples}={fid} "
                f"Gen_steps={args.gen_step} "
                f"Conf={args.conf_temp} "
                f"Samp={args.samp_temp} "
                f"Pre={args.pre_full_iters} "
                f"Cache={args.num_cache_iters} "
                f"Cfg={args.cfg}\n"
            )


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    args = get_args()
    config = OmegaConf.load(args.config)
    decode(config, args)
