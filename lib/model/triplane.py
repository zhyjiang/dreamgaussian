
import math

import torch
import torch.nn as nn
from einops import rearrange, repeat



class TriplaneLearnablePositionalEmbedding(nn.Module):
    def __init__(self, plane_size: int = 32, num_channels: int = 1024):
        super().__init__()
        self.plane_size = plane_size
        self.num_channels = num_channels
        self.embeddings = nn.Parameter(
            torch.randn(
                (3, self.num_channels, self.plane_size, self.plane_size),
                dtype=torch.float32,
            )
            * 1
            / math.sqrt(self.num_channels)
        )
   
    def forward(self, batch_size, cond_embeddings = None) :
        import ipdb; ipdb.set_trace()
        embeddings = repeat(self.embeddings, "Np Ct Hp Wp -> B Np Ct Hp Wp", B=batch_size)
        if cond_embeddings is not None:
            embeddings = embeddings + cond_embeddings
        return rearrange(
            embeddings,
            "B Np Ct Hp Wp -> B Ct (Np Hp Wp)",
        )

    def detokenize(
        self, tokens
    ):
        batch_size, Ct, Nt = tokens.shape
        assert Nt == self.cfg.plane_size**2 * 3
        assert Ct == self.cfg.num_channels
        return rearrange(
            tokens,
            "B Ct (Np Hp Wp) -> B Np Ct Hp Wp",
            Np=3,
            Hp=self.cfg.plane_size,
            Wp=self.cfg.plane_size,
        )
