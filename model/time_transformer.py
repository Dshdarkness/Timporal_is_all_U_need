import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class RandomPatchPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # batch,channel,16,7x7
        b, c, t, h, w = x.shape
        x = x.reshape(b, c, t, h * w)
        if self.training and my_cfg.model.transformer.random_select:
            while True:
                idx = random.randint(0, h * w - 1)
                i = idx // h
                j = idx % h
                if j == 0 or i == h - 1 or j == h - 1:
                    continue
                else:
                    break
        else:
            idx = h * w // 2
        x = x[..., idx]
        return x

def valid_idx(idx, h):
    i = idx // h
    j = idx % h
    if j == 0 or i == h - 1 or j == h - 1:
        return False
    else:
        return True

class RandomAvgPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # batch,channel,16,7x7
        b, c, t, h, w = x.shape
        x = x.reshape(b, c, t, h * w)
        candidates = list(range(h * w))
        candidates = [idx for idx in candidates if valid_idx(idx, h)]
        max_k = len(candidates)
        if self.training and my_cfg.model.transformer.random_select:
            k = my_cfg.model.transformer.k
        else:
            k = max_k
        candidates = random.sample(candidates, k)
        x = x[..., candidates].mean(-1)
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, mask = None):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape #batch,num_patches,channels  #

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)



def valid_idx(idx, h):
    i = idx // h
    j = idx % h
    pad = h // 7
    if j < pad or i >= h - pad or j >= h - pad:
        return False
    else:
        return True

import random
from config import config as my_cfg
from math import sqrt
class RandomSelect(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # batch,7x7
        size=x.shape[1]
        h=int(sqrt(size))
        candidates = list(range(size))
        candidates = [idx for idx in candidates if valid_idx(idx, h)]
        max_k = len(candidates)
        if self.training and my_cfg.model.transformer.random_select:
            k = my_cfg.model.transformer.k
            if k==-1:
                k=max_k
        else:
            k = max_k
        candidates = random.sample(candidates, k)
        x = x[:,candidates]
        return x

class VideoiT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_patches, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch = Rearrange('b c t (h p1) (w p2) -> b (h w) t (p1 p2 c)', p1 = patch_size, p2 = patch_size)
        self.patch_to_embedding=nn.Linear(patch_dim, dim)
        self.num_patches=num_patches
        

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.random_select=RandomSelect()
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, mask = None):
        real_b=img.shape[0]
        x = self.to_patch(img)
        x = self.random_select(x)
        n=x.shape[1]
        x=x.reshape(real_b*n,self.num_patches,-1)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape #batch,num_patches,channels  #

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)
        x = x.reshape(real_b,-1)
        return x


class TimeTransformer(nn.Module):
    def __init__(self,num_patches, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.num_patches=num_patches
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        b, n, _ = x.shape #batch,num_patches,channels  #

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask=None)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class TransformerHead(nn.Module):
    def __init__(self, spatial_size=7, time_size=16, in_channels=2048):
        super().__init__()
        if my_cfg.model.inco.no_time_pool:
            time_size = time_size * 2
        patch_type = my_cfg.model.transformer.patch_type
        if patch_type == "time":
            self.pool = nn.AvgPool3d((1, spatial_size, spatial_size))
            self.num_patches = time_size
        elif patch_type == "spatial":
            self.pool = nn.AvgPool3d((time_size, 1, 1))
            self.num_patches = spatial_size ** 2
        elif patch_type == "random":
            self.pool = RandomPatchPool()
            self.num_patches = time_size
        elif patch_type == "random_avg":
            self.pool = RandomAvgPool()
            self.num_patches = time_size
        elif patch_type == "all":
            self.pool = nn.Identity()
            self.num_patches = time_size * spatial_size * spatial_size
        else:
            raise NotImplementedError(patch_type)

        self.dim = my_cfg.model.transformer.dim
        if self.dim == -1:
            self.dim = in_channels
            my_cfg.model.transformer.dim = self.dim

        self.in_channels = in_channels

        if self.dim != self.in_channels:
            self.fc = nn.Linear(self.in_channels, self.dim)

        default_params = dict(
            dim=self.dim, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1,
        )
        params = my_cfg.model.transformer.to_dict()
        for key in default_params:
            if key in params:
                default_params[key] = params[key]
        self.time_T = TimeTransformer(
            num_patches=self.num_patches, num_classes=1, **default_params
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(x[0])
        x = x.reshape(-1, self.in_channels, self.num_patches)
        x = x.permute(0, 2, 1)
        if self.dim != self.in_channels:
            x = self.fc(x.reshape(-1, self.in_channels))
            x = x.reshape(-1, self.num_patches, self.dim)
        x = self.time_T(x)
        x = self.sigmoid(x)
        return x
