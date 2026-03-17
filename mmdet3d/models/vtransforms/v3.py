from typing import Tuple, List
import torch
from torch import nn
import torch.nn.functional as F
import einops
import torch.utils.checkpoint as cp
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from ..utils.reference_poinst import get_reference_points
__all__ = ["BEVTransformV3"]
import torch
import torch.nn.functional as F

class DualCoordAttentionGateV3(nn.Module):

    def __init__(self, inp, oup, groups=32, reduction=8, use_residual=True):
        super(DualCoordAttentionGateV3, self).__init__()
        self.use_residual = use_residual

        self.pool_h_mean = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_mean = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))

        mip = max(8, inp // groups)

        self.shared_conv1 = nn.Conv2d(inp, mip, kernel_size=1, bias=False)
        self.shared_bn = nn.GroupNorm(num_groups=4, num_channels=mip)
        self.relu = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, bias=False)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inp, inp // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp // reduction, 2, 1),
            nn.Softmax(dim=1)
        )

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(oup, oup // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(oup // reduction, oup, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h_mean = self.pool_h_mean(x)
        x_w_mean = self.pool_w_mean(x).permute(0, 1, 3, 2)
        y_mean = torch.cat([x_h_mean, x_w_mean], dim=2)

        y_mean = self.shared_conv1(y_mean)
        y_mean = self.shared_bn(y_mean)
        y_mean = self.relu(y_mean)

        x_h_mean, x_w_mean = torch.split(y_mean, [h, w], dim=2)
        x_w_mean = x_w_mean.permute(0, 1, 3, 2)
        attn_mean = self.conv_h(x_h_mean).sigmoid() * self.conv_w(x_w_mean).sigmoid()

        x_h_max = self.pool_h_max(x)
        x_w_max = self.pool_w_max(x).permute(0, 1, 3, 2)
        y_max = torch.cat([x_h_max, x_w_max], dim=2)

        y_max = self.shared_conv1(y_max)
        y_max = self.shared_bn(y_max)
        y_max = self.relu(y_max)

        x_h_max, x_w_max = torch.split(y_max, [h, w], dim=2)
        x_w_max = x_w_max.permute(0, 1, 3, 2)
        attn_max = self.conv_h(x_h_max).sigmoid() * self.conv_w(x_w_max).sigmoid()

        gate_weights = self.gate(identity)
        attn = attn_mean * gate_weights[:, 0:1] + attn_max * gate_weights[:, 1:2]

        out = identity * attn
        scale = self.channel_att(out)
        out = out * scale

        if self.use_residual:
            out = out + identity
        return out


class ResidualRefinementV3(BaseModule):

    def __init__(self, channels):
        super().__init__()
        mid_channels = channels // 2

        self.conv1 = nn.Conv3d(channels, mid_channels,
                               kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=mid_channels)
        self.relu = nn.ReLU(True)

        self.attention = DualCoordAttentionGateV3(mid_channels, mid_channels, reduction=8)

        self.conv2 = nn.Conv3d(mid_channels, mid_channels,
                               kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=mid_channels)

        self.conv_out = nn.Conv3d(mid_channels, channels, kernel_size=1, bias=False)
        self.gn_out = nn.GroupNorm(num_groups=8, num_channels=channels)

        self.init_weights()

    def init_weights(self):
        for m in [self.conv1, self.conv2, self.conv_out]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        for m in self.attention.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.gn_out.weight, 0)
        nn.init.constant_(self.gn_out.bias, 0)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        B, C, Z, H, W = out.shape
        out_reshaped = einops.rearrange(out, 'b c z h w -> (b z) c h w')
        out_att = self.attention(out_reshaped)
        out = einops.rearrange(out_att, '(b z) c h w -> b c z h w', b=B, z=Z)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        out = self.conv_out(out)
        out = self.gn_out(out)

        return identity + out


@MODELS.register_module()
class BEVTransformV3(BaseModule):
    def __init__(
            self,
            x: List[float], y: List[float], z: List[float],
            xs: int, ys: int, zs: int,
            input_size: List[int],
            in_channels: int = 256,
            out_channels: int = 128,
            top_type: str = 'lidar',
            down_sample: bool = False,
            down_sample_scale: int = 2,
            down_sample_channels: List[int] = [128 * 10, 64 * 10, 32 * 10, 16 * 10],
            with_cp: bool = False,
            init_cfg=None,
    ):
        super(BEVTransformV3, self).__init__(init_cfg=init_cfg)
        self.pc_range = [x[0], y[0], z[0], x[1], y[1], z[1]]
        self.volume_size = [int(s) for s in [xs, ys, zs]]
        ref_3d = get_reference_points(self.pc_range[0], self.pc_range[3],
                                      self.pc_range[1], self.pc_range[4],
                                      self.pc_range[2], self.pc_range[5],
                                      self.volume_size[0], self.volume_size[1], self.volume_size[2])
        self.register_buffer('ref_3d', ref_3d)

        if in_channels != out_channels:
            self.transfer_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.transfer_conv = None

        self.out_channels = out_channels
        self.input_size = input_size
        self.top_type = top_type
        self.with_cp = with_cp
        self.refinement_module = ResidualRefinementV3(out_channels)

        if down_sample:
            assert down_sample_scale in [1, 2]
            if down_sample_scale == 2:
                self.down_sample = nn.Sequential(
                    nn.Conv2d(down_sample_channels[0], down_sample_channels[1], 3, padding=1, bias=False),
                    nn.BatchNorm2d(down_sample_channels[1]),
                    nn.ReLU(True),
                    nn.Conv2d(down_sample_channels[1], down_sample_channels[2], 3, stride=down_sample_scale, padding=1,
                              bias=False),
                    nn.BatchNorm2d(down_sample_channels[2]),
                    nn.ReLU(True),
                    nn.Conv2d(down_sample_channels[2], down_sample_channels[3], 3, padding=1, bias=False),
                    nn.BatchNorm2d(down_sample_channels[3]),
                    nn.ReLU(True),
                )
            else:
                self.down_sample = nn.Sequential(
                    nn.Conv2d(down_sample_channels[0], down_sample_channels[1], 3, padding=1, bias=False),
                    nn.BatchNorm2d(down_sample_channels[1]),
                    nn.ReLU(True)
                )
        else:
            self.down_sample = None

    def init_weights(self):
        super().init_weights()
        if hasattr(self, 'refinement_module'):
            self.refinement_module.init_weights()

        if self.transfer_conv is not None and not hasattr(self.transfer_conv, '_is_init'):
            nn.init.normal_(self.transfer_conv.weight, std=0.01)

    def forward_sampling_sum(self, x, reference_points_img, volume_mask):
        B, C, N, H, W = x.shape
        num_points = reference_points_img.shape[2]

        feat_accum = torch.zeros((B, num_points, C), device=x.device, dtype=torch.float32)

        for n in range(N):
            x_view = x[:, :, n, :, :]
            ref_view = reference_points_img[:, n, :, :]
            mask_view = volume_mask[:, n, :, :]

            valid_index_per_batch = [mask_view[i].squeeze(-1).nonzero().squeeze(-1) for i in range(B)]
            max_len = max([len(idx) for idx in valid_index_per_batch])

            if max_len == 0:
                continue

            grid = x.new_zeros([B, max_len, 1, 2])
            for b_i in range(B):
                idx = valid_index_per_batch[b_i]
                if len(idx) > 0:
                    grid[b_i, :len(idx), 0, :] = ref_view[b_i, idx, :]

            grid = 2 * grid - 1

            # Grid Sample
            feat_sample = F.grid_sample(x_view, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            feat_sample = feat_sample.squeeze(-1).permute(0, 2, 1)

            # 2. FP32 累加
            for b_i in range(B):
                idx = valid_index_per_batch[b_i]
                if len(idx) > 0:
                    valid_feat = feat_sample[b_i, :len(idx), :]
                    feat_accum[b_i, idx, :] += valid_feat.float()

        feat_accum = torch.clamp(feat_accum, min=-30.0, max=30.0)
        return feat_accum.to(x.dtype)

    def forward(self, x, points, camera2ego, lidar2ego, lidar2camera, lidar2image, camera_intrinsics,
                camera2lidar, img_aug_matrix, lidar_aug_matrix, metas, **kwargs):
                
                
        if x.dtype != torch.float32: x = x.float()
        B = x.shape[0]

        if self.transfer_conv is not None:
            x = einops.rearrange(x, 'B N C H W -> (B N) C H W')
            x = self.transfer_conv(x)
            x = einops.rearrange(x, '(B N) C H W -> B C N H W', B=B)
        else:
            x = einops.rearrange(x, 'B N C H W -> B C N H W')

        if self.top_type == 'ego':
            assert 'camera_ego2global' in kwargs
            camera_ego2global = kwargs['camera_ego2global']
            keyego2global = camera_ego2global[:, 0, ...].unsqueeze(1)
            global2keyego = torch.inverse(keyego2global.double())
            camera2sensor = global2keyego @ camera_ego2global.double() @ camera2ego.double()
            camera2sensor = camera2sensor.float()
        elif self.top_type == 'lidar':
            camera2sensor = camera2lidar
        else:
            raise NotImplementedError

        reference_points_img, volume_mask = self.point_sampling(camera2sensor, camera_intrinsics[..., :3, :3],
                                                                img_aug_matrix[..., :3, :3], img_aug_matrix[..., :3, 3],
                                                                lidar_aug_matrix)

        if self.with_cp and x.requires_grad:
            feats_volume_flat = cp.checkpoint(self.forward_sampling_sum, x, reference_points_img, volume_mask,
                                              use_reentrant=False)
        else:
            feats_volume_flat = self.forward_sampling_sum(x, reference_points_img, volume_mask)

        feats_3d = einops.rearrange(feats_volume_flat,
                                    'b (z h w) c -> b c z h w',
                                    z=self.volume_size[2],
                                    h=self.volume_size[0],
                                    w=self.volume_size[1])

        if self.with_cp and feats_3d.requires_grad:
            feats_3d = cp.checkpoint(self.refinement_module, feats_3d, use_reentrant=False)
        else:
            feats_3d = self.refinement_module(feats_3d)

        feats_out = einops.rearrange(feats_3d, 'b c z h w -> b (z c) w h')

        if self.down_sample is not None:
            feats_out = self.down_sample(feats_out)

        return feats_out

    def point_sampling(self, camera2sensor, cam2imgs, post_rots, post_trans, bda, mode='fix'):
        if camera2sensor.dtype != torch.float32: camera2sensor = camera2sensor.float()
        if cam2imgs.dtype != torch.float32: cam2imgs = cam2imgs.float()
        if bda.dtype != torch.float32: bda = bda.float()
        if post_rots.dtype != torch.float32: post_rots = post_rots.float()
        if post_trans.dtype != torch.float32: post_trans = post_trans.float()

        B, N, _, _ = camera2sensor.shape
        num_points = self.ref_3d.shape[0]

        reference_points = self.ref_3d.view(1, num_points, 3) - bda[:, :3, 3].view(B, 1, 3)
        reference_points = torch.inverse(bda[:, :3, :3].view(B, 1, 3, 3)).matmul(reference_points.unsqueeze(-1))
        reference_points = reference_points.view(B, 1, num_points, 3, 1).squeeze(-1)
        reference_points = (reference_points - camera2sensor[:, :, :3, 3].view(B, N, 1, 3)).unsqueeze(-1)
        combine = cam2imgs.matmul(camera2sensor[:, :, :3, :3].transpose(-1, -2))
        reference_points_img = combine.view(B, N, 1, 3, 3).matmul(reference_points).squeeze(-1)

        eps = 1e-5
        volume_mask = (reference_points_img[..., 2:3] > eps)
        reference_points_img = reference_points_img[..., 0:2] / torch.maximum(
            reference_points_img[..., 2:3], torch.ones_like(reference_points_img[..., 2:3]) * eps)

        post_rots2 = post_rots[:, :, :2, :2]
        post_trans2 = post_trans[:, :, :2]
        reference_points_img = post_rots2.view(B, N, 1, 2, 2).matmul(reference_points_img.unsqueeze(-1))
        reference_points_img = reference_points_img.squeeze(-1) + post_trans2.view(B, N, 1, 2)

        H_in, W_in = self.input_size
        reference_points_img[..., 0] /= W_in
        reference_points_img[..., 1] /= H_in

        volume_mask = (volume_mask & (reference_points_img[..., 1:2] > 0.0)
                       & (reference_points_img[..., 1:2] < 1.0)
                       & (reference_points_img[..., 0:1] < 1.0)
                       & (reference_points_img[..., 0:1] > 0.0))

        volume_mask = torch.nan_to_num(volume_mask)

        return reference_points_img, volume_mask
