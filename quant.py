"""Quantization layers for CryoCodex, adapted from FoundationVision/VAR.
Copyright (c) 2024 FoundationVision.
MIT license; see THIRD_PARTY_NOTICES.md for full terms and attribution.
"""

from typing import List, Tuple

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F


def Normalize(in_channels, num_groups=32, affine=True):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=affine)


class VectorQuantizer3(nn.Module):
    def __init__(
        self,
        vocab_size=4096,
        Cvae=256,
        using_znorm=False,
        beta: float = 0.25,
        default_qresi_counts=0,
        v_patch_nums=None,
        quant_resi=0.5,
        decay=0.99,
        patience=128,
        gamma=1.2,
        n_heads=8,
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.Cvae: int = Cvae
        self.n_heads: int = n_heads
        assert Cvae % n_heads == 0, f"Cvae {Cvae} must be divisible by n_heads {n_heads}"
        self.head_dim = Cvae // n_heads

        self.using_znorm: bool = using_znorm
        self.v_patch_nums: Tuple[int] = v_patch_nums
        self.norm1 = Normalize(Cvae)

        self.quant_resi_ratio = quant_resi
        phi_dim = self.head_dim
        self.quant_resi = PhiNonShared([(Phi(phi_dim, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(default_qresi_counts or len(self.v_patch_nums))])
        self.gamma = gamma
        self.record_hit = 0

        self.beta: float = beta
        self.embedding = nn.Embedding(self.vocab_size, self.head_dim)

        self.prog_si = -1

        self.decay = decay
        self.eps = 1e-5
        self.embedding.weight.requires_grad = False
        self.patience_time = patience
        self.register_buffer('patience', torch.full((self.vocab_size,), self.patience_time, dtype=torch.long))
        self.register_buffer('ema_cluster_size', torch.zeros(self.vocab_size))
        self.register_buffer('ema_w', torch.zeros(self.vocab_size, self.head_dim))
        self.register_buffer('avg_usage', torch.zeros(self.vocab_size))

    def eini(self, eini):
        if eini > 0:
            nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0:
            self.embedding.weight.data.uniform_(-abs(eini) / self.vocab_size, abs(eini) / self.vocab_size)

    def extra_repr(self) -> str:
        return f'{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}, heads={self.n_heads} | S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}'

    def forward(self, f_BChwd: torch.Tensor, mask: torch.Tensor = None, ret_usages=True, return_index=False) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        import torch.distributed as dist
        dtype = f_BChwd.dtype
        f_BChwd = self.norm1(f_BChwd)
        if dtype != torch.float32:
            f_BChwd = f_BChwd.float()

        B, C, H, W, D = f_BChwd.shape
        f_no_grad = f_BChwd.detach()

        f_rest = f_no_grad.view(B, self.n_heads, self.head_dim, H, W, D).reshape(-1, self.head_dim, H, W, D).clone()
        f_hat = torch.zeros_like(f_rest)

        mask_expanded = None
        if mask is not None:
            mask_expanded = mask.repeat_interleave(self.n_heads, dim=0)
        if self.training:
            total_dw = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BChwd.device)
            total_dw_embed = torch.zeros(self.vocab_size, self.head_dim, dtype=torch.float, device=f_BChwd.device)

        last_scale_indices = None
        mean_vq_loss: torch.Tensor = 0.0
        SN = len(self.v_patch_nums)
        C_head = self.head_dim

        for si, pn in enumerate(self.v_patch_nums):
            if si != SN - 1:
                f_interp = F.interpolate(f_rest, size=(pn, pn, pn), mode='area')
                rest_NC = f_interp.permute(0, 2, 3, 4, 1).reshape(-1, C_head)
            else:
                rest_NC = f_rest.permute(0, 2, 3, 4, 1).reshape(-1, C_head)

            if self.using_znorm:
                rest_NC = F.normalize(rest_NC, dim=-1)
                idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
                idx_N = torch.argmin(d_no_grad, dim=1)

            if self.training:
                encodings = F.one_hot(idx_N, self.vocab_size).float()
                dw = encodings.sum(0)
                total_dw.add_(dw)
                dw_embed = encodings.T @ rest_NC
                total_dw_embed.add_(dw_embed)

            idx_Bhwd = idx_N.view(B * self.n_heads, pn, pn, pn)
            z_q = self.embedding(idx_Bhwd).permute(0, 4, 1, 2, 3)

            if si != SN - 1:
                h_BChwd = F.interpolate(z_q, size=(H, W, D), mode='trilinear').contiguous()
            else:
                h_BChwd = z_q.contiguous()
                if return_index:
                    last_scale_indices = idx_N.view(B * self.n_heads, -1)
            h_BChwd = self.quant_resi[si / (SN - 1)](h_BChwd)

            f_hat = f_hat + h_BChwd
            f_rest -= h_BChwd

            target_split = f_no_grad.view(B, self.n_heads, self.head_dim, H, W, D).reshape(-1, self.head_dim, H, W, D)
            squared_diff = (target_split - f_hat.detach()) ** 2

            if mask_expanded is not None:
                weighted_diff = squared_diff * mask_expanded
                dims = (1, 2, 3, 4)
                per_sample_error_sum = weighted_diff.sum(dim=dims)
                per_sample_weight_sum = mask_expanded.sum(dim=dims) * squared_diff.shape[1]
                per_sample_loss = per_sample_error_sum / (per_sample_weight_sum + 1e-6)
                current_loss = per_sample_loss.mean()
            else:
                current_loss = squared_diff.mean()

            mean_vq_loss += current_loss.mul_(self.beta)
        if self.training:
            if dist.is_initialized():
                dist.all_reduce(total_dw, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_dw_embed, op=dist.ReduceOp.SUM)
            global_effective_B = B * self.n_heads
            if dist.is_initialized():
                global_effective_B *= dist.get_world_size()
            self.avg_usage.mul_(self.decay).add_((total_dw / global_effective_B).mul(0.1))
            self.ema_cluster_size.data.mul_(self.decay).add_(total_dw, alpha=1 - self.decay)
            self.ema_w.data.mul_(self.decay).add_(total_dw_embed, alpha=1 - self.decay)

            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + self.eps) / (n + self.vocab_size * self.eps) * n
            embed_normalized = self.ema_w / cluster_size.unsqueeze(1)

            if self.using_znorm:
                embed_normalized = F.normalize(embed_normalized, dim=1)
            self.embedding.weight.data.copy_(embed_normalized)

            used_mask = total_dw > 0
            self.patience[used_mask] = self.patience_time
            self.patience[~used_mask] -= 1

        mean_vq_loss *= 1. / SN

        f_hat_restored = f_hat.detach().view(B, self.n_heads, self.head_dim, H, W, D).reshape(B, C, H, W, D)
        f_hat_final = (f_hat_restored - f_no_grad).add_(f_BChwd)
        if ret_usages:
            usages = (self.avg_usage > 0.1).float().mean().item()
        else:
            usages = None

        if return_index and last_scale_indices is not None:
            return f_hat_final, usages, mean_vq_loss * 0.25, last_scale_indices
        return f_hat_final, usages, mean_vq_loss * 0.25


class Phi(nn.Conv3d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks // 2)
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BChwd):
        return h_BChwd.mul(1 - self.resi_ratio) + super().forward(h_BChwd).mul_(self.resi_ratio)


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        K = len(qresi)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'
