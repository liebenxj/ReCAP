from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.diffloss import DiffLoss
import time
from timm.layers import Mlp
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List



def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking




class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True


        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, past_kv=None, return_kv=False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if past_kv is not None:
            k = torch.cat((past_kv[0], k), dim=2)
            v = torch.cat((past_kv[1], v), dim=2)


        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_kv:
            return x, (k, v)
        else:
            return x



class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, past_kv=None, return_kv=False) -> torch.Tensor:
        attn = self.attn(self.norm1(x), past_kv=past_kv, return_kv=return_kv)
        if return_kv:
            attn, (k, v) = attn
        x = x + self.drop_path1(self.ls1(attn))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        if return_kv:
            return x, (k, v)
        else:
            return x

class MAR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.grad_checkpointing = grad_checkpointing

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # MAR decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def forward_mae_encoder(self, x, mask, class_embedding, past_kvs=None, return_kv=False, cache_buffer=False, position_ids=None):
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape
        if past_kvs is None:
            past_kvs = [None] * len(self.encoder_blocks)

        # concat buffer
        if not cache_buffer:
            x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
            mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)
        else:
            mask_with_buffer = mask

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        if not cache_buffer:
            x[:, :self.buffer_size] = class_embedding.unsqueeze(1)

        # encoder position embedding
        if position_ids is not None:
            x = x + self.encoder_pos_embed_learned[0, position_ids]
        else:
            x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # dropping
        x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        # apply Transformer blocks
        if return_kv:
            current_kvs = []
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x)
        else:
            for i, block in enumerate(self.encoder_blocks):
                x = block(x, past_kv=past_kvs[i], return_kv=return_kv)
                if return_kv:
                    x, current_kv = x
                    current_kvs.append(current_kv)
        x = self.encoder_norm(x)
        
        if return_kv:
            return x, current_kvs
        else:
            return x

    def forward_mae_decoder(self, x, mask, past_kvs=None, return_kv=False, cache_buffer=False, position_ids=None):
        if past_kvs is None:
            past_kvs = [None] * len(self.decoder_blocks)

        x = self.decoder_embed(x)
        if not cache_buffer:
            mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)
        else:
            mask_with_buffer = mask

     
        # pad mask tokens
        
        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        # decoder position embedding
        if position_ids is not None:
            x = x_after_pad + self.decoder_pos_embed_learned[0, position_ids]
        else:
            x = x_after_pad + self.decoder_pos_embed_learned

        # apply Transformer blocks
        if return_kv:
            current_kvs = []
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for i, block in enumerate(self.decoder_blocks):
                x = block(x, past_kv=past_kvs[i], return_kv=return_kv)
                if return_kv:
                    x, current_kv = x
                    current_kvs.append(current_kv)

        x = self.decoder_norm(x)

        if not cache_buffer:
            x = x[:, self.buffer_size:]
        if position_ids is not None:
            position_ids = position_ids - self.buffer_size
            if not cache_buffer:
                position_ids = position_ids[:,self.buffer_size:]
            x = x + self.diffusion_pos_embed_learned[0, position_ids]
        else:
            x = x + self.diffusion_pos_embed_learned
        if return_kv:
            return x, current_kvs
        else:
            return x

    def forward_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def forward(self, imgs, labels):

        # class embed
        class_embedding = self.class_emb(labels)

        # patchify and mask (drop) tokens
        x = self.patchify(imgs)
        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz=x.size(0))
        mask = self.random_masking(x, orders)

        # mae encoder
        x = self.forward_mae_encoder(x, mask, class_embedding)

        # mae decoder
        z = self.forward_mae_decoder(x, mask)

        # diffloss
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss
    


    def extract_kvs(self, past_kv, mask_to_pred, cache_buffer=False):
        bsz, num_heads, emb_dim = past_kv.shape[0], past_kv.shape[1],past_kv.shape[-1]
        past_kv_ = past_kv[:,:,self.buffer_size:].transpose(1,2)[~mask_to_pred].reshape(bsz,-1,num_heads,emb_dim).transpose(1,2)
        if cache_buffer:
            past_kv_ = torch.cat((past_kv[:,:,:self.buffer_size], past_kv_), dim=2)
        return past_kv_
        


    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False, pre_full_iters=0, num_cache_iters=1, cache_buffer=True, diff_fast_ratio=0.5):

        # init and sample generation orders
        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        orders = self.sample_orders(bsz)
        diff_time = 0
        diff_z = []


        iter_cnt = 0
        while iter_cnt < num_iter:
            cur_tokens = tokens.clone()
            # class embedding and CFG
            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)
            if not cfg == 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask = torch.cat([mask, mask], dim=0)

            # mae encoder
            x, enc_past_kvs = self.forward_mae_encoder(tokens, mask, class_embedding, return_kv=True)

            # mae decoder
            z, dec_past_kvs = self.forward_mae_decoder(x, mask, return_kv=True)

   
            num_masked_tokens = torch.sum(mask, dim=-1)[0].item()
            # if num_masked_tokens < 160:
            #     num_cache_iters = 2

            mask_lens = []
            for i in range(num_cache_iters*(iter_cnt>pre_full_iters-1)+1):
                ratio = 1. * (iter_cnt + 1 + i) / num_iter
                mask_ratio = np.cos(math.pi / 2. * ratio)
                mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()
                # Keeps at least one of prediction in this round and also masks out at least
                # one and for the next iteration
                if i == 0:
                    minimum_mask_len = torch.sum(mask, dim=-1, keepdims=True) - 1
                else:
                    minimum_mask_len = mask_lens[-1] - 1
                mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                    torch.minimum(minimum_mask_len, mask_len))
                if ratio == 1.0:
                    mask_len = torch.Tensor([0]).cuda()
                    mask_lens.append(mask_len[0])
                    break
                else:
                    mask_lens.append(mask_len[0])
                


            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_lens[-1], orders, bsz, self.seq_len)
            

            if iter_cnt + num_cache_iters*(iter_cnt>pre_full_iters-1) >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)
            

            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_lens[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            

            if num_cache_iters*(iter_cnt>pre_full_iters-1) == 0:
                z = z[mask_to_pred.nonzero(as_tuple=True)]
                torch.cuda.synchronize()
                start = time.time()
                diff_z.append(z.shape[0])
                sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
                torch.cuda.synchronize()
                diff_time += time.time() - start

                if not cfg == 1.0:
                    sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0) 
                
                

            
            if num_cache_iters*(iter_cnt>pre_full_iters-1) > 0:
                ids_to_pred = torch.where(mask_to_pred)[1].reshape(len(mask_to_pred), -1)
                n_pred_ids = ids_to_pred.shape[1]
                batch_range = torch.arange(ids_to_pred.shape[0]).reshape(-1, 1).repeat(1, n_pred_ids)    

                ntoken_schedule = (torch.tensor(mask_lens[:-1]) - torch.tensor(mask_lens[1:])).long().tolist()
                ntoken_schedule = [n_pred_ids-sum(ntoken_schedule)] + ntoken_schedule  


                local_mask = torch.ones(bsz, n_pred_ids).cuda()
                local_tokens = torch.zeros(bsz, n_pred_ids, self.token_embed_dim).cuda()

                
                
                
                local_position_ids = ids_to_pred + self.buffer_size

                if not cache_buffer:
                    local_position_ids = torch.cat([torch.arange(self.buffer_size).cuda().unsqueeze(0).repeat(local_position_ids.shape[0], 1), local_position_ids], dim=1)
                    local_enc_past_kvs = []
                    for past_kv in enc_past_kvs:
                        local_enc_past_kvs.append((past_kv[0][:,:,self.buffer_size:], past_kv[1][:,:,self.buffer_size:]))
                else:
                    local_enc_past_kvs = enc_past_kvs
                    

                local_dec_past_kvs = []
                for past_kv in dec_past_kvs:
                    local_dec_past_kvs.append((self.extract_kvs(past_kv[0], mask_to_pred, cache_buffer=cache_buffer), self.extract_kvs(past_kv[1], mask_to_pred, cache_buffer=cache_buffer)))
                

                
                s_idx = ntoken_schedule[0]
                z = z[batch_range,ids_to_pred][:,:s_idx].reshape(-1, z.shape[-1])
                torch.cuda.synchronize()
                start = time.time()
                diff_z.append(z.shape[0])
                sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
                torch.cuda.synchronize()
                diff_time += time.time() - start
                if not cfg == 1.0:
                    sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0) 
                

                sampled_token_latent = sampled_token_latent.reshape(bsz, -1, self.token_embed_dim)
                local_tokens[:, :s_idx] = sampled_token_latent[:, :s_idx]
                iter_cnt += 1
                local_mask[:, :s_idx] = 0

                for step, ntoken in enumerate(ntoken_schedule[1:]):
                    e_idx = s_idx + ntoken
                    cur_local_tokens = local_tokens.clone()
                    if not cfg == 1.0:
                        local_tokens = torch.cat([local_tokens, local_tokens], dim=0)
                        local_mask = torch.cat([local_mask, local_mask], dim=0)

                    local_x = self.forward_mae_encoder(local_tokens, local_mask, class_embedding, past_kvs=local_enc_past_kvs, cache_buffer=cache_buffer, position_ids=local_position_ids)
                    
                    local_z = self.forward_mae_decoder(local_x, local_mask, past_kvs=local_dec_past_kvs, cache_buffer=cache_buffer, position_ids=local_position_ids)
                    local_z = local_z[:, s_idx:e_idx].reshape(-1, z.shape[-1])
                    if cfg_schedule == "linear":
                        cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_lens[step+1]) / self.seq_len
                    elif cfg_schedule == "constant":
                        cfg_iter = cfg
                    torch.cuda.synchronize()
                    start = time.time()
                    diff_z.append(local_z.shape[0])
                    sampled_token_latent = self.diffloss.sample(local_z, temperature, cfg_iter, fast=True, fast_ratio=diff_fast_ratio)
                    torch.cuda.synchronize()
                    diff_time += time.time() - start
                    if not cfg == 1.0:
                        sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class 
                    cur_local_tokens[:, s_idx:e_idx] = sampled_token_latent.reshape(bsz, -1, self.token_embed_dim)
                    local_tokens = cur_local_tokens.clone()
                    local_mask[:, s_idx:e_idx] = 0
                    if not cfg == 1.0:
                        local_mask, _ = local_mask.chunk(2, dim=0)

                    s_idx = e_idx
                    iter_cnt += 1
                    

                if not cfg == 1.0:
                    ids_to_pred, _ = ids_to_pred.chunk(2, dim=0)
                cur_tokens[batch_range[:bsz],ids_to_pred] = local_tokens
                
            else:
                if not cfg == 1.0:
                    mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)
                cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
                iter_cnt += 1


            tokens = cur_tokens.clone()




        # unpatchify
        tokens = self.unpatchify(tokens)
        return tokens, diff_time


def mar_base(**kwargs):
    model = MAR(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_large(**kwargs):
    model = MAR(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mar_huge(**kwargs):
    model = MAR(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
