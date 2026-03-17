from typing import List
import warnings
import torch
import einops
import torch.nn.functional as F
from torch import nn
import numpy as np
from mmengine.model import BaseModule
from mmcv.cnn import build_norm_layer
from mmengine.dist import get_dist_info
from mmdet.registry import MODELS
from ..utils.reference_poinst import get_reference_points

__all__ = ["CrossCoordinateSample"]

@MODELS.register_module()
class CrossCoordinateSample(BaseModule):
    # TODO: [yz] make CrossCoordinateSample more generalized
    def __init__(self, point_range: list, point_num: list, lidar_point_range: list, point_type: str = 'ego',
                 extra_up: bool = False, extra_up_scale: int = 2, in_dim: int = 512, out_dim: int = 128,
                 norm_cfg=dict(type='BN'), init_cfg=None) -> None:  # [新增] init_cfg
        super().__init__(init_cfg=init_cfg)
        ref_points = get_reference_points(*point_range, *point_num)
        self.register_buffer('ref_points', ref_points)
        if len(lidar_point_range) == 4:
            warnings.warn("4-element lidar_point_range is deprecated, please use 6-element format")
            self.lidar_y_min, self.lidar_y_max, self.lidar_x_min, self.lidar_x_max = lidar_point_range
        else:
            assert len(lidar_point_range) == 6
            self.lidar_x_min, self.lidar_y_min, _, self.lidar_x_max, self.lidar_y_max, _ = lidar_point_range
        self.output_h, self.output_w = point_num[:2]
        assert point_type in ['ego', 'lidar']
        self.point_type = point_type
        if in_dim != out_dim:
            self.transfer_conv = torch.nn.Conv2d(in_dim, out_dim, kernel_size=1)
        else:
            self.transfer_conv = None
        self.extra_up = extra_up
        if extra_up:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=extra_up_scale, mode='bilinear', align_corners=True),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_dim)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_dim, out_dim, kernel_size=1, padding=0)
            )
        self.fp16_enabled = False

    def forward(self, x: torch.Tensor,
                lidar_aug_matrix: torch.Tensor, lidar2ego: torch.Tensor, occ_aug_matrix: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32: x = x.float()
        if lidar_aug_matrix is not None and lidar_aug_matrix.dtype != torch.float32:
            lidar_aug_matrix = lidar_aug_matrix.float()
        if lidar2ego is not None and lidar2ego.dtype != torch.float32:
            lidar2ego = lidar2ego.float()
        if occ_aug_matrix is not None and occ_aug_matrix.dtype != torch.float32:
            occ_aug_matrix = occ_aug_matrix.float()

        if self.transfer_conv is not None:
            x = self.transfer_conv(x)
        batch_size = x.shape[0]
        if lidar_aug_matrix is None:
            lidar_aug_matrix = torch.eye(4, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(batch_size, 1, 1)

        B = lidar_aug_matrix.shape[0]
        num_points = self.ref_points.shape[0]
        ref_points_fp32 = self.ref_points.float() if self.ref_points.dtype != torch.float32 else self.ref_points
        if self.point_type == 'ego':
            # inverse occ data augment
            # ([B 3 3] -> [B 1 3 3]) @ ([num_points 3] -> [1 num_points 3 1]) -> [B num_points 3 1]
            ref_ego = torch.inverse(occ_aug_matrix[:, :3, :3].view(B, 1, 3, 3)).matmul(
                ref_points_fp32.view(1, num_points, 3, 1))
            # [B num_points 3 1] -> [B num_points 3 1] -> [B num_points 3]
            ref_ego = ref_ego.view(B, num_points, 3, 1).squeeze(-1)

            # ego to lidar
            # [Safety Check] 防止 lidar2ego 为 None
            if lidar2ego is None:
                lidar2ego = torch.eye(4, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(batch_size, 1, 1)

            # [B num_points 3] - ([B 3] -> [B 1 3]) -> [B num_points 3] -> [B num_points 3 1]
            ref_lidar = (ref_ego - lidar2ego[:, :3, 3].view(batch_size, 1, 3)).unsqueeze(-1)
            # ([B 3 3] -> [B 1 3 3]) @ [B num_points 3 1] -> [B num_points 3 1]
            ref_lidar = torch.inverse(lidar2ego[:, :3, :3]).view(batch_size, 1, 3, 3).matmul(ref_lidar)

            # lidar data augment
            # ([B 3 3] -> [B 1 3 3]) @ [B num_points 3 1] -> [B num_points 3 1]
            ref_lidar = lidar_aug_matrix[:, :3, :3].view(B, 1, 3, 3).matmul(ref_lidar.view(B, num_points, 3, 1))
            # ([B num_points 3 1] -> [B num_points 3]) + ([B 3] -> [B 1 3]) -> [B num_points 3]
            ref_lidar = ref_lidar.squeeze(-1) + lidar_aug_matrix[:, :3, 3].unsqueeze(1)
            # [B num_points 3] -> [B num_points 1 3] -> [B num_points 1 2]
            ref_lidar = ref_lidar.unsqueeze(-2)[:, :, :, :2]
        else:  # self.point_type == 'lidar'
            ref_lidar = ref_points_fp32[:, :2].unsqueeze(-2).unsqueeze(0).repeat(B, 1, 1, 1)
        lidar_x_length, lidar_y_length = self.lidar_x_max - self.lidar_x_min, self.lidar_y_max - self.lidar_y_min
        ref_lidar[..., 0] = (ref_lidar[..., 0] - self.lidar_x_min) / lidar_x_length
        ref_lidar[..., 1] = (ref_lidar[..., 1] - self.lidar_y_min) / lidar_y_length
        ref_lidar = ref_lidar * 2 - 1
        rank, _ = get_dist_info()

        x = F.grid_sample(
            x,
            ref_lidar,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        x = einops.rearrange(x.squeeze(-1), 'bs c (h w) -> bs c h w', h=self.output_h, w=self.output_w)
        if self.extra_up:
            x = self.up(x)
        return x
