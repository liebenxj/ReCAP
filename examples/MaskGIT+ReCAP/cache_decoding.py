import datetime
import math
import os
import time
import argparse
import json
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm
import accelerate
import einops
from omegaconf import OmegaConf
from loguru import logger
import models.vqgan
from models.uvit import UViT
from torch._C import _distributed_c10d
import sys
from einops import rearrange
from copy import deepcopy
import torch.nn.functional as F
import cv2




_distributed_c10d.set_debug_level(_distributed_c10d.DebugLevel.INFO)





def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/maskgit.yaml',)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--benchmark', type=int, default=0)
    parser.add_argument('--test_bsz', type=int, default=64)
    parser.add_argument('--gen_step', type=int, default=8)
    parser.add_argument('--cfg', action='store_true', default=False)
    parser.add_argument('--pre_full_iters', type=int, default=0)
    parser.add_argument('--num_cache_iters', type=int, default=0)
    parser.add_argument('--eval_n', type=int, default=50000)
    parser.add_argument('--samp_temp', type=float, default=0.7)
    parser.add_argument('--conf_temp', type=float, default=5.5)
    args = parser.parse_args()
    return args



@logger.catch()
def decode(config, args):
    logger.add(os.path.join(args.output_dir, 'output.log'), level='INFO')

    if args.benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(args.seed, device_specific=True)

    if accelerator.is_main_process:
        logger.info('Setting seed: {}'.format(args.seed))
    logger.info(f'Process {accelerator.process_index} using device: {device}')
    logger.info(f'Using mini-batch size {args.test_bsz} per device')



    accelerator.wait_for_everyone()
    logger.info(f'Run on {accelerator.num_processes} devices')

    
    autoencoder = models.vqgan.get_model()
    codebook_size = autoencoder.n_embed
    config.nnet.codebook_size = codebook_size
    autoencoder.to(device)

    @torch.cuda.amp.autocast(enabled=True)
    def encode(_batch):
        return autoencoder.encode(_batch)[-1][-1].reshape(len(_batch), -1)

    @torch.cuda.amp.autocast(enabled=True)
    def decode(_batch):
        return autoencoder.decode_code(_batch)

    # for cfg:
    empty_ctx = torch.from_numpy(np.array([[1000]], dtype=np.longlong)).to(device)
    nnet_ema = UViT(**config.nnet)
    nnet_ema = accelerator.prepare(nnet_ema)
    ckpt = torch.load(config.nnet.pretrained_path, map_location='cpu')
    try:
        nnet_ema.module.load_state_dict(ckpt)
    except:
        nnet_ema.load_state_dict(ckpt)
    nnet_ema.eval()



    def cfg_nnet(x, labels, scale, cond_past_kvs=None,uncond_past_kvs=None, use_cache=False, position_ids=None):
        _cond, cond_past_kvs = nnet_ema(x, context=labels, past_kvs=cond_past_kvs, use_cache=use_cache, position_ids=position_ids)
        _uncond, uncond_past_kvs = nnet_ema(x, context=einops.repeat(empty_ctx, '1 ... -> B ...', B=x.size(0)), past_kvs=uncond_past_kvs, use_cache=use_cache, position_ids=position_ids)
        res = _cond + scale * (_cond - _uncond)
        return res, cond_past_kvs, uncond_past_kvs




    def amortize(n_samples, batch_size):
        k = n_samples // batch_size
        r = n_samples % batch_size
        return k * [batch_size] if r == 0 else k * [batch_size] + [r]

    def get_data_generator():
        while True:
            yield torch.randint(0, 1000, (args.test_bsz, 1), device=device)
    data_generator = get_data_generator()
    n_samples = args.eval_n
    batch_size = args.test_bsz * accelerator.num_processes



    mask_ind = codebook_size
    seq_len = 256
    hw = 16


    def add_gumbel_noise(t, temperature, device):
        return (t + torch.Tensor(temperature * np.random.gumbel(size=t.shape)).to(device))






    ratio_schedule = 1 - np.linspace(0, 1, args.gen_step+1)[:-1]**2.5
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
                for i in range(args.num_cache_iters):
                    if cnt + i < len(ntoken_schedule):
                        local_ntoken_schedule.append(ntoken_schedule[cnt + i])
                        cnt += 1
                    else:
                        break
            local_ntoken_schedules.append(local_ntoken_schedule)
        ntoken_schedule = [sum(i) for i in local_ntoken_schedules]
    assert sum(ntoken_schedule) == seq_len




    samp_temp_schedule = args.samp_temp + (1 - np.linspace(0, 1, args.gen_step)**0.5)*(1.0 - args.samp_temp)
    samp_temp_schedule = samp_temp_schedule.tolist()
    conf_temp_schedule = args.conf_temp * (1. - np.array([(i+1) / len(ntoken_schedule) for i in range(len(ntoken_schedule))]))
    cfg_schedule = np.linspace(0.05, 0.1, args.gen_step).tolist()
    
    
    local_samp_temp_schedules = []
    local_cfg_schedules = []
    for i in local_ntoken_schedules:
        local_samp_temp_schedules.append(samp_temp_schedule[:len(i)])
        local_cfg_schedules.append(cfg_schedule[:len(i)])
        samp_temp_schedule = samp_temp_schedule[len(i):]
        cfg_schedule = cfg_schedule[len(i):]

    
    if not args.cfg:
        cfg_schedule = None



    save_folder = os.path.join(args.output_dir, f'step{args.gen_step}_conf{args.conf_temp}_samp{args.samp_temp}_pre{args.pre_full_iters}_cache{args.num_cache_iters}')
    os.makedirs(save_folder, exist_ok=True)

    idx = 0
    cnt = 0
    for _ in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc=f'sample'):
        with torch.no_grad():
            labels = next(data_generator)
            nnet_ = cfg_nnet if args.cfg else nnet_ema
            x = torch.full((labels.shape[0], seq_len), mask_ind, dtype=torch.long).to(device)
            for step, ntoken in enumerate(ntoken_schedule):
                batch_range = torch.arange(labels.shape[0]).unsqueeze(-1).expand(-1, ntoken)
                is_mask = x.eq(mask_ind)
                if args.cfg:
                    local_cfg_schedule = local_cfg_schedules[step]
                    cfg_scale = local_cfg_schedule[0]
                    logits, full_cond_past_kvs, full_uncond_past_kvs = nnet_(x, labels, scale=cfg_scale, cond_past_kvs=None, uncond_past_kvs=None, use_cache=True)
                else:
                    logits, full_past_kvs = nnet_(x, context=labels, past_kvs=None, use_cache=True)


                local_ntoken_schedule = local_ntoken_schedules[step]
                local_gen_step = len(local_ntoken_schedule)
                local_samp_temp_schedule = local_samp_temp_schedules[step]

                    
                ####confidence
                x_ = torch.distributions.Categorical(logits=logits).sample()
                conf_temp = conf_temp_schedule[step]
                logits_ = torch.log_softmax(logits, dim=-1)
                logits_ = torch.gather(logits_, dim=-1, index=x_.unsqueeze(-1)).squeeze(-1)
                logits_ = torch.where(is_mask, logits_, -np.inf).float()
                confidence = add_gumbel_noise(logits_, conf_temp, device)
                sorted_confidence, sorted_idx = torch.sort(confidence, dim=-1, descending=True)
                unmask_idx = sorted_idx[:, :ntoken]
                cached_idx = sorted_idx[:, ntoken:]

                if len(local_ntoken_schedule) == 1:
                    x[batch_range, unmask_idx] = x_[batch_range, unmask_idx]
                    continue

                local_x = torch.full((labels.shape[0], ntoken), mask_ind, dtype=torch.long).to(device)
                local_logits = logits[batch_range, unmask_idx]  
                batch_range1 = torch.arange(labels.shape[0]).unsqueeze(-1).expand(-1, seq_len-ntoken)
                if args.cfg:
                    cond_past_kvs = []
                    uncond_past_kvs = []
                    for cond_past_kv in full_cond_past_kvs:
                        cond_past_kvs.append((cond_past_kv[0][:,1:][batch_range1, cached_idx], cond_past_kv[1][:,1:][batch_range1, cached_idx]))
                    for uncond_past_kv in full_uncond_past_kvs:
                        uncond_past_kvs.append((uncond_past_kv[0][:,1:][batch_range1, cached_idx], uncond_past_kv[1][:,1:][batch_range1, cached_idx]))
                else:
                    past_kvs = []
                    for past_kv in full_past_kvs:
                        past_kvs.append((past_kv[0][:,1:][batch_range1, cached_idx], past_kv[1][:,1:][batch_range1, cached_idx]))
     
                
                s_idx = 0
                for local_step, local_ntoken in enumerate(local_ntoken_schedule):
                    local_temp = local_samp_temp_schedule[local_step]
                    e_idx = s_idx + local_ntoken
                    local_x[:, s_idx:e_idx] = torch.distributions.Categorical(logits=local_logits/local_temp).sample()[:, s_idx:e_idx]
                    if local_step < local_gen_step - 1:
                        if args.cfg:
                            cfg_scale = local_cfg_schedule[local_step]
                            local_logits, _, _ = nnet_(local_x, labels, scale=cfg_scale, cond_past_kvs=cond_past_kvs, uncond_past_kvs=uncond_past_kvs, use_cache=False, position_ids=unmask_idx)
                        else:
                            local_logits, _ = nnet_(local_x, context=labels, past_kvs=past_kvs, use_cache=False, position_ids=unmask_idx)
                    s_idx = e_idx  
                assert local_x.eq(mask_ind).sum() == 0
                x[batch_range, unmask_idx] = local_x
                        


        assert x.eq(mask_ind).sum() == 0
        _z = rearrange(x, 'b (i j) -> b i j', i=hw, j=hw)
        gen_images_batch = decode(_z).detach().cpu()
        for b_id in range(gen_images_batch.size(0)):
            img_id = cnt * gen_images_batch.size(0) * accelerator.num_processes + accelerator.process_index * gen_images_batch.size(0) + b_id
            if img_id >= n_samples:
                break
            gen_img = np.clip(gen_images_batch[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255)
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)
        idx = idx + x.shape[0]
        cnt += 1




if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    args = get_args()
    config = OmegaConf.load(args.config)
    decode(config, args)

