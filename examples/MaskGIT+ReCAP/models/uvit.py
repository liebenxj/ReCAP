import torch
import torch.nn as nn

import timm
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import PatchEmbed, Mlp

assert timm.__version__ == "0.3.2"  # version check
import einops
import torch.utils.checkpoint
from loguru import logger




if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, vocab_size, hidden_size, max_position_embeddings, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        torch.nn.init.normal_(self.word_embeddings.weight, std=.02)
        torch.nn.init.normal_(self.position_embeddings.weight, std=.02)

    def forward(
            self, input_ids, position_ids=None
    ):  
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1), device=input_ids.device).long().unsqueeze(0)
        position_embeddings = self.position_embeddings(position_ids)
        

        inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MlmLayer(nn.Module):

    def __init__(self, feat_emb_dim, word_emb_dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(feat_emb_dim, word_emb_dim)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(word_emb_dim)
        self.bias = nn.Parameter(torch.zeros(1, 1, vocab_size))

    def forward(self, x, word_embeddings):
        mlm_hidden = self.fc(x)
        mlm_hidden = self.gelu(mlm_hidden)
        mlm_hidden = self.ln(mlm_hidden)
        word_embeddings = word_embeddings.transpose(0, 1)
        logits = torch.matmul(mlm_hidden, word_embeddings)
        logits = logits + self.bias
        return logits


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, past_kv=None, use_cache=False):
        B, N, C = x.shape

        if ATTENTION_MODE == 'flash':
            qkv = self.qkv(x)
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            if past_kv is not None:
                k = torch.cat([past_kv[0], k], dim=2)
                v = torch.cat([past_kv[1], v], dim=2)
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')

        elif ATTENTION_MODE == 'xformers':
            qkv = self.qkv(x)
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            if past_kv is not None:
                k = torch.cat([past_kv[0], k], dim=1)
                v = torch.cat([past_kv[1], v], dim=1)
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
            
        elif ATTENTION_MODE == 'math':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            if past_kv is not None:
                # ForkedPdb().set_trace()
                k = torch.cat([past_kv[0], k], dim=2)
                v = torch.cat([past_kv[1], v], dim=2)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        present_kv = None
        if use_cache:
            present_kv = (k, v)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, present_kv


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None, past_kv=None, use_cache=False):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip, past_kv, use_cache)

    def _forward(self, x, skip=None, past_kv=None, use_cache=False):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        # x = x + self.attn(self.norm1(x))
        x_norm1 = self.norm1(x)
        attn_out, present_kv = self.attn(x_norm1, past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out

        x = x + self.mlp(self.norm2(x))


        return x, present_kv


class UViT(nn.Module):
    def __init__(self, img_size=16, in_chans=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, use_checkpoint=False, pretrained_path=None,
                 skip=True, codebook_size=1024, num_classes=None):
        super().__init__()
        logger.debug(f'codebook size in nnet: {codebook_size}')
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.in_chans = in_chans
        self.skip = skip

        self.codebook_size = codebook_size
        vocab_size = codebook_size + 1
        self.time_embed = None
        self.num_vis_tokens = int((img_size) ** 2)
        self.token_emb = BertEmbeddings(vocab_size=vocab_size,
                                        hidden_size=embed_dim,
                                        max_position_embeddings=self.num_vis_tokens,
                                        dropout=0.1)
        print(f'num vis tokens: {self.num_vis_tokens}')
        self.txt_encoder = None


        # conditioning
        self.extras = 1
        self.context_embed = BertEmbeddings(vocab_size=num_classes,
                                            hidden_size=embed_dim,
                                            max_position_embeddings=1,
                                            dropout=0)

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.mlm_layer = MlmLayer(feat_emb_dim=embed_dim, word_emb_dim=embed_dim, vocab_size=vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, masked_ids, timesteps=None, context=None, past_kvs=None, use_cache=False, position_ids=None):
        if past_kvs is None:
            past_kvs = [None] * len(self.in_blocks) + [None] + [None] * len(self.out_blocks)
        new_past_kvs = []

        assert len(masked_ids.shape) == 2
        x = self.token_emb(masked_ids, position_ids=position_ids)
        context_token = self.context_embed(context)
        x = torch.cat((context_token, x), dim=1)

        if self.skip:
            skips = []
        for i, blk in enumerate(self.in_blocks):
            x, present_kv = blk(x, past_kv=past_kvs[i], use_cache=use_cache)
            new_past_kvs.append(present_kv)
            if self.skip:
                skips.append(x)
        x, present_kv = self.mid_block(x, past_kv=past_kvs[len(self.in_blocks)], use_cache=use_cache)
        new_past_kvs.append(present_kv)

        for j, blk in enumerate(self.out_blocks):
            if self.skip:
                x, present_kv = blk(x, skip=skips.pop(),past_kv=past_kvs[len(self.in_blocks) + 1 + j], use_cache=use_cache)
            else:
                x, present_kv = blk(x, past_kv=past_kvs[len(self.in_blocks) + 1 + j], use_cache=use_cache)
            new_past_kvs.append(present_kv)

                                                        
        x = self.norm(x)
        word_embeddings = self.token_emb.word_embeddings.weight.data.detach()
        x = self.mlm_layer(x, word_embeddings)
        x = x[:, self.extras:, :self.codebook_size]
        if use_cache:
            return x, new_past_kvs
        else:
            return x, None
  
