import torch.nn as nn
import torch
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .utils import get_1d_sincos_pos_embed


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

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class VertexTransformer(nn.Module):
    def __init__(self, hidden_dim=64, num_joints=6890, num_layers=3, pose_dim=2, nhead=4, dropout=0.1,
                 dim_head=64, mlp_dim=64, has_bbox=False,upsample=1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_joints = num_joints
        self.pose_dim = pose_dim
        
        self.positional_emb = nn.Parameter(torch.randn((1, num_joints, hidden_dim)))
        # self.cls_token = nn.Parameter(torch.randn((hidden_dim)))
        self.proj_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.pre_emb = nn.Linear(pose_dim, hidden_dim)
        self.pre_norm = nn.LayerNorm(hidden_dim)
        
        self.encoder = Transformer(hidden_dim, num_layers, nhead, dim_head, mlp_dim, dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.upsample_conv = nn.Conv1d(num_joints, num_joints*upsample, kernel_size=1) if upsample!=1 else None
        self.has_bbox = has_bbox
        if has_bbox:
            self.bbox_tokenize = nn.Linear(4, hidden_dim, bias=False)
            
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.upsample = upsample
        self.initialize_weights()
        
        self.mean3D_head = self._make_head(hidden_dim, 3)
        self.opacity_head = self._make_head(hidden_dim, 1)
        self.shs_head = self._make_head(hidden_dim, 3)
        self.rotations_head = self._make_head(hidden_dim, 4)
        self.scales_head = self._make_head(hidden_dim, 3)
    
    def _make_head(self, hidden_dim, out_dim):
        layers = nn.Sequential(nn.Linear(self.hidden_dim, hidden_dim),
                               nn.GELU(),
                               nn.Dropout(0.1),
                               nn.Linear(hidden_dim, out_dim))
        return layers    
    
    def initialize_weights(self):
        pos_embed = get_1d_sincos_pos_embed(self.positional_emb.shape[-1], np.arange(self.num_joints))
        self.positional_emb.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv1d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x, mask_ratio=0.0):
        mask = None
        if x.shape[-1] != self.pose_dim:
            mask = x[:, :, -1] == 0
            x = x[:, :, :-1]
        x = self.pre_emb(x)
        x = self.pre_norm(x)
        if mask is not None:
            x = self.masking(x, mask)
        # x = torch.cat((self.cls_token.repeat(x.shape[0], 1, 1), x), dim=1)
        x = x + self.positional_emb
        x = self.dropout(x)
        
        x = self.encoder(x)
        
        if self.upsample != 1:
            x = self.upsample_conv(x)
        # import ipdb; ipdb.set_trace()
        # x = self.proj_layer(x)
        
        means3D = self.mean3D_head(x)
        # import ipdb; ipdb.set_trace()
        opacity = self.opacity_head(x).sigmoid()
        shs = self.shs_head(x)
        rotations = self.rotations_head(x).sigmoid()
        scales = self.scales_head(x).sigmoid()
        return means3D, opacity, scales, shs, rotations
    
    def masking(self, x, mask):
        x[mask] = self.mask_token
        return x
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore).bool()
        x[mask] = self.mask_token

        return x, mask, ids_restore
