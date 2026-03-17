import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from typing import List, Union, Tuple

__all__ = ["SpectralFusion"]

def vector_to_grid(x_vec):
    B, C, _, _ = x_vec.shape
    Hc = int(math.floor(math.sqrt(C)))
    Wc = int(math.ceil(C / Hc))
    pad = Hc * Wc - C
    if pad > 0:
        x_vec = F.pad(x_vec.view(B, C), (0, pad))
    grid = x_vec.view(B, 1, Hc, Wc)
    return grid, (Hc, Wc, C, pad)


def grid_to_vector(grid, meta):
    Hc, Wc, C, pad = meta
    B = grid.size(0)
    vec = grid.view(B, Hc * Wc)
    if pad > 0:
        vec = vec[:, :C]
    return vec.view(B, C, 1, 1)


class LiteAmplitudeGating(BaseModule):

    def __init__(self, dropout_rate=0.1, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.amp_mlp = nn.Sequential(
            nn.Conv2d(1, 4, 1, bias=False),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(4, 1, 1, bias=True),  # 开启 Bias
            nn.Sigmoid()
        )
    def init_weights(self):
        super().init_weights()
        nn.init.xavier_normal_(self.amp_mlp[0].weight)
        nn.init.constant_(self.amp_mlp[3].weight, 0)
        nn.init.constant_(self.amp_mlp[3].bias, -10.0)
    def forward(self, spec):
        amp = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)
        dtype = self.amp_mlp[0].weight.dtype
        filter_mask = self.amp_mlp(amp.to(dtype))
        spec_modulated = spec * filter_mask.float()
        return spec_modulated

@MODELS.register_module()
class SpectralFusion(BaseModule):
    def __init__(self,
                 in_channels: Union[int, List[int]],
                 out_channels: int,
                 dropout: float = 0.1,
                 use_checkpoint: bool = True,
                 init_cfg: dict = None):
        super().__init__(init_cfg=init_cfg)

        cam_c = in_channels[0] if isinstance(in_channels, list) else in_channels
        lidar_c = in_channels[1] if isinstance(in_channels, list) else in_channels
        self.use_checkpoint = use_checkpoint
        self.cam_proj = nn.Sequential(
            nn.Conv2d(cam_c, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(True)
        )
        self.lidar_proj = nn.Sequential(
            nn.Conv2d(lidar_c, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(True)
        )
        self.base_fuser = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(True)
        )
        self.spectral_gate = LiteAmplitudeGating(dropout_rate=dropout)
        self.gamma = nn.Parameter(torch.tensor(0.2), requires_grad=True)
    def init_weights(self):
        super().init_weights()

        for m in [self.cam_proj, self.lidar_proj, self.base_fuser]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

        self.spectral_gate.init_weights()

    def _get_spectral_weights(self, x):
        chan_desc = F.adaptive_avg_pool2d(x, 1)
        if chan_desc.dtype == torch.float32:
            chan_desc = torch.nan_to_num(chan_desc)
        grid, meta = vector_to_grid(chan_desc)
        spec = torch.fft.fft2(grid.float())
        spec_modulated = self.spectral_gate(spec)
        grid_ifft = torch.fft.ifft2(spec_modulated).real
        weight_vec = grid_to_vector(grid_ifft, meta)
        att_weights = torch.tanh(weight_vec)
        return att_weights
    def forward(self, inputs: Union[Tuple[torch.Tensor], List[torch.Tensor]]) -> torch.Tensor:
        cam, lidar = inputs
        cam = self.cam_proj(cam)
        lidar = self.lidar_proj(lidar)
        x_cat = torch.cat([cam, lidar], dim=1)
        x_base = self.base_fuser(x_cat)
        if self.use_checkpoint and self.training and x_base.requires_grad:
            att_weights = cp.checkpoint(self._get_spectral_weights, x_base, use_reentrant=False)
        else:
            att_weights = self._get_spectral_weights(x_base)
        clean_weights = torch.nan_to_num(att_weights.to(x_base.dtype))
        out = x_base + self.gamma * (x_base * clean_weights)
        return out
