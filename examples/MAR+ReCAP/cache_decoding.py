import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


from models.vae import AutoencoderKL
from models import mar
import copy
import cv2


def get_args_parser():
    parser = argparse.ArgumentParser('MAR', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='mar_large', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--grad_checkpointing', action='store_true')

    # VAE parameters
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--vae_path', default="pretrained_models/vae/kl16.ckpt", type=str,
                        help='images input size')
    parser.add_argument('--vae_embed_dim', default=16, type=int,
                        help='vae output embedding dimension')
    parser.add_argument('--vae_stride', default=16, type=int,
                        help='tokenizer stride, default use KL16')
    parser.add_argument('--patch_size', default=1, type=int,
                        help='number of tokens to group as a patch.')

    # Generation parameters
    parser.add_argument('--num_iter', default=64, type=int,
                        help='number of autoregressive iterations to generate an image')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='number of images to generate')
    parser.add_argument('--cfg', default=1.0, type=float, help="classifier-free guidance")
    parser.add_argument('--cfg_schedule', default="linear", type=str)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--eval_bsz', type=int, default=64, help='generation batch size')



    # MAR params
    parser.add_argument('--mask_ratio_min', type=float, default=0.7,
                        help='Minimum mask ratio')
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clip')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--proj_dropout', type=float, default=0.1,
                        help='projection dropout')
    parser.add_argument('--buffer_size', type=int, default=64)

    # Diffusion Loss params
    parser.add_argument('--diffloss_d', type=int, default=12)
    parser.add_argument('--diffloss_w', type=int, default=1536)
    parser.add_argument('--num_sampling_steps', type=str, default="100")
    parser.add_argument('--diff_fast_ratio', type=str, default=1.0)
    parser.add_argument('--diffusion_batch_mul', type=int, default=1)
    parser.add_argument('--temperature', default=1.0, type=float, help='diffusion loss sampling temperature')


    parser.add_argument('--class_num', default=1000, type=int)
    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)


    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # ReCAP parameters
    parser.add_argument('--cache_buffer', action='store_true', default=False)
    parser.add_argument('--pre_full_iters', default=0, type=int)
    parser.add_argument('--num_cache_iters', default=1, type=int)

    return parser



def evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=16, cfg=1.0,
             use_ema=True):
    model_without_ddp.eval()
    num_steps = args.num_images // (batch_size * misc.get_world_size()) + 1
    save_folder = os.path.join(args.output_dir, "step{}-diffstep{}-temp{}-{}cfg{}-image{}-pre{}-cache{}-buffer{}".format(args.num_iter, args.num_sampling_steps,args.temperature,args.cfg_schedule,cfg, args.num_images, args.pre_full_iters, args.num_cache_iters, args.cache_buffer))
    if use_ema:
        save_folder = save_folder + "_ema"
    print("Save to:", save_folder)
    if misc.get_rank() == 0:
        os.makedirs(save_folder, exist_ok=True)
  
             
            
    # switch to ema params
    if use_ema:
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)

    class_num = args.class_num
    assert args.num_images % class_num == 0  # number of images per class must be the same
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    used_time = 0
    used_diff_time = 0
    gen_img_cnt = 0

    if num_steps > 0:
        for i in range(num_steps):
            print("Generation step {}/{}".format(i, num_steps))

            labels_gen = class_label_gen_world[world_size * batch_size * i + local_rank * batch_size:
                                                    world_size * batch_size * i + (local_rank + 1) * batch_size]
            labels_gen = torch.Tensor(labels_gen).long().cuda()


            torch.cuda.synchronize()
            start_time = time.time()

            # generation
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    sampled_tokens, diff_time = model_without_ddp.sample_tokens(bsz=batch_size, num_iter=args.num_iter, cfg=cfg, cfg_schedule=args.cfg_schedule, labels=labels_gen,temperature=args.temperature, pre_full_iters=args.pre_full_iters, num_cache_iters=args.num_cache_iters, cache_buffer=args.cache_buffer, diff_fast_ratio=args.diff_fast_ratio)
                    sampled_images = vae.decode(sampled_tokens / 0.2325)

            # measure speed after the first generation batch
            if i >= 1:
                used_diff_time += diff_time
                torch.cuda.synchronize()
                used_time += time.time() - start_time
                gen_img_cnt += batch_size
                print("Generating {} images takes {:.5f} seconds, {:.5f} sec per image (diff:{:.5f})".format(gen_img_cnt, used_time, used_time / gen_img_cnt, used_diff_time / gen_img_cnt))

            torch.distributed.barrier()
            sampled_images = sampled_images.detach().cpu()
            sampled_images = (sampled_images + 1) / 2

            # distributed save
            for b_id in range(sampled_images.size(0)):
                img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
                if img_id >= args.num_images:
                    break
                gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
                gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
                cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

        torch.distributed.barrier()
        time.sleep(10)




def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)



    # define the vae and mar model
    vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path).cuda().eval()
    for param in vae.parameters():
        param.requires_grad = False

    model = mar.__dict__[args.model](
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        mask_ratio_min=args.mask_ratio_min,
        label_drop_prob=args.label_drop_prob,
        class_num=args.class_num,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        buffer_size=args.buffer_size,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=args.num_sampling_steps,
        diffusion_batch_mul=args.diffusion_batch_mul,
        grad_checkpointing=args.grad_checkpointing,
    )

    print("Model = %s" % str(model))
    # following timm: set wd as 0 for bias and norm layers
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))

    model.to(device)
    model_without_ddp = model



    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module



    assert os.path.exists(os.path.join(args.resume, "checkpoint-last.pth"))
    checkpoint = torch.load(os.path.join(args.resume, "checkpoint-last.pth"), map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])
    model_params = list(model_without_ddp.parameters())
    ema_state_dict = checkpoint['model_ema']
    ema_params = [ema_state_dict[name].cuda() for name, _ in model_without_ddp.named_parameters()]
    print("Resume checkpoint")


    torch.cuda.empty_cache()
    evaluate(model_without_ddp, vae, ema_params, args, 0, batch_size=args.eval_bsz, cfg=args.cfg, use_ema=True)
        
 

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)
