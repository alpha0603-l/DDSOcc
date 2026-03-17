import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import math
from typing import Tuple

from mmdet.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class UnboundedOccBEVTransform(BaseModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            image_size: Tuple[int, int],
            feature_size: Tuple[int, int],
            xbound: Tuple[float, float, float],
            ybound: Tuple[float, float, float],
            zbound: Tuple[float, float, float],
            dbound: Tuple[float, float, float],
            downsample: int = 1,
            R_core: float = 30.0,
            alpha: float = 0.6,
            num_rho: int = 256,
            num_theta: int = 512,
            num_z: int = 10,
            align_corners: bool = True,
            use_checkpoint: bool = True,
            **kwargs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound
        self.downsample_rate = downsample

        self.use_checkpoint = use_checkpoint
        self.align_corners = align_corners

        dx = max(abs(xbound[0]), abs(xbound[1]))
        dy = max(abs(ybound[0]), abs(ybound[1]))
        self.R_max = 55.0

        self.R_core = float(R_core)
        self.alpha = float(alpha)

        self.num_rho = int(num_rho)
        self.num_theta = int(num_theta)
        self.num_z = int(num_z)

        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1),
        )

        self.polar_encoder = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, bias=False),  # Padding 手动处理
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
        )

        self.z_compression = nn.Sequential(
            nn.Conv2d(out_channels * self.num_z, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if downsample > 1:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, stride=downsample, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample_layer = nn.Identity()

        self._init_grids()

    def _init_grids(self):
        x_min, x_max, x_step = self.xbound
        y_min, y_max, y_step = self.ybound
        self.bev_xs = int(round((x_max - x_min) / x_step))
        self.bev_ys = int(round((y_max - y_min) / y_step))

        bev_x_lin = torch.linspace(x_min + x_step / 2, x_max - x_step / 2, self.bev_xs)
        bev_y_lin = torch.linspace(y_min + y_step / 2, y_max - y_step / 2, self.bev_ys)
        self.register_buffer("bev_x_lin", bev_x_lin, persistent=False)
        self.register_buffer("bev_y_lin", bev_y_lin, persistent=False)

        rho_lin = torch.linspace(0.0, 1.0, self.num_rho)
        theta_lin = torch.linspace(-math.pi, math.pi, self.num_theta)
        z_lin = torch.linspace(self.zbound[0], self.zbound[1], self.num_z)

        self.register_buffer("rho_lin", rho_lin, persistent=False)
        self.register_buffer("theta_lin", theta_lin, persistent=False)
        self.register_buffer("z_lin", z_lin, persistent=False)

        param_grid = self._build_param_grid_xyz().float()
        self.register_buffer("param_grid", param_grid, persistent=False)

    def rho_to_radius(self, rho):
        R_core, R_max, alpha = self.R_core, self.R_max, self.alpha
        r = torch.empty_like(rho)
        mask = rho < alpha
        if mask.any():
            r[mask] = rho[mask] * (R_core / alpha)
        if (~mask).any():
            r[~mask] = R_core + (R_max - R_core) * (rho[~mask] - alpha) / (1 - alpha + 1e-6)
        return r

    def _build_param_grid_xyz(self):
        rho, theta, z = self.rho_lin, self.theta_lin, self.z_lin
        r = self.rho_to_radius(rho)
        zz, tt, rr = torch.meshgrid(z, theta, r, indexing="ij")
        xx = rr * torch.cos(tt)
        yy = rr * torch.sin(tt)
        return torch.stack([xx, yy, zz], dim=-1)

    def get_cam_feats(self, x):
        B, N, C, fH, fW = x.shape
        x = x.view(B * N, C, fH, fW)
        x = self.depthnet(x)
        return x

    def _inner_view_transform(self, cam_feats_2d, lidar_aug_matrix, camera2ego, camera_intrinsics, img_aug_matrix):
        B, N, C, H_feat, W_feat = cam_feats_2d.shape
        feat_dtype = cam_feats_2d.dtype
        device = cam_feats_2d.device

        cam_feats_flatten = cam_feats_2d.view(B * N, C, H_feat, W_feat)

        param_grid = self.param_grid.to(device=device)
        Zp, Tp, Rp, _ = param_grid.shape
        P = Zp * Tp * Rp

        pts_ego = param_grid.reshape(1, 1, P, 3).expand(B, N, P, 3).clone()

        aug_rot = lidar_aug_matrix[:, :3, :3].float()
        aug_trans = lidar_aug_matrix[:, :3, 3].float()
        aug_rot_inv = torch.inverse(aug_rot)

        pts_ego = pts_ego - aug_trans.view(B, 1, 1, 3)
        pts_ego = torch.matmul(aug_rot_inv.view(B, 1, 1, 3, 3), pts_ego.unsqueeze(-1)).squeeze(-1)

        rots = camera2ego[..., :3, :3].float()
        trans = camera2ego[..., :3, 3].float()
        rots_inv = rots.transpose(-1, -2)

        pts_cam = torch.matmul(rots_inv.unsqueeze(2), (pts_ego - trans.unsqueeze(2)).unsqueeze(-1)).squeeze(-1)
        x_c, y_c, z_c = pts_cam[..., 0], pts_cam[..., 1], pts_cam[..., 2]

        z_c_safe = torch.clamp(z_c, min=1e-5)
        fx = camera_intrinsics[..., 0, 0][:, :, None].float()
        fy = camera_intrinsics[..., 1, 1][:, :, None].float()
        cx = camera_intrinsics[..., 0, 2][:, :, None].float()
        cy = camera_intrinsics[..., 1, 2][:, :, None].float()

        u = (x_c * fx) / z_c_safe + cx
        v = (y_c * fy) / z_c_safe + cy

        uv = torch.stack([u, v, torch.ones_like(u)], dim=-1)
        img_aug_rot = img_aug_matrix[..., :3, :3].unsqueeze(2).float()
        img_aug_trans = img_aug_matrix[..., :3, 3].unsqueeze(2).float()
        uv_aug = torch.matmul(img_aug_rot, uv.unsqueeze(-1)).squeeze(-1) + img_aug_trans
        u_aug, v_aug = uv_aug[..., 0], uv_aug[..., 1]

        u_norm = (u_aug / (self.image_size[1] - 1)) * 2 - 1
        v_norm = (v_aug / (self.image_size[0] - 1)) * 2 - 1

        in_img_mask = (u_norm >= -1) & (u_norm <= 1) & (v_norm >= -1) & (v_norm <= 1)
        valid_mask_all = (z_c > 0.5) & in_img_mask

        chunk_size = 80000
        num_chunks = (P + chunk_size - 1) // chunk_size
        polar_vol_flat_list = []

        u_norm_flat = u_norm.view(B, N, P)
        v_norm_flat = v_norm.view(B, N, P)
        valid_mask_flat = valid_mask_all.view(B, N, P)

        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, P)

            u_chunk = u_norm_flat[..., start:end]
            v_chunk = v_norm_flat[..., start:end]
            mask_chunk = valid_mask_flat[..., start:end]

            u_chunk_clone = u_chunk.clone()
            v_chunk_clone = v_chunk.clone()
            u_chunk_clone[~mask_chunk] = 2.0
            v_chunk_clone[~mask_chunk] = 2.0

            grid_chunk = torch.stack([u_chunk_clone, v_chunk_clone], dim=-1)
            grid_chunk = grid_chunk.view(B * N, end - start, 1, 2)
            grid_chunk_low = grid_chunk.to(feat_dtype)

            feat_chunk = F.grid_sample(cam_feats_flatten, grid_chunk_low,
                                       align_corners=self.align_corners, padding_mode="zeros")

            feat_chunk = feat_chunk.view(B, N, self.out_channels, end - start)
            mask_chunk_low = mask_chunk.view(B, N, 1, end - start).to(feat_dtype)

            sum_chunk = (feat_chunk * mask_chunk_low).sum(dim=1)
            overlap_count = torch.clamp(mask_chunk_low.sum(dim=1), min=1.0)
            polar_vol_flat_list.append(sum_chunk / overlap_count)

        polar_vol_flat = torch.cat(polar_vol_flat_list, dim=-1)
        polar_vol = polar_vol_flat.view(B, self.out_channels, Zp, Tp, Rp)
        polar_vol_spatial = polar_vol.permute(0, 2, 1, 3, 4).reshape(B * Zp, self.out_channels, Tp, Rp)
        x_pad_rho = F.pad(polar_vol_spatial, (1, 1, 0, 0), mode='constant', value=0)
        top_row = x_pad_rho[..., -1:, :]
        bottom_row = x_pad_rho[..., :1, :]
        x_padded = torch.cat([top_row, x_pad_rho, bottom_row], dim=-2)
        polar_vol_encoded = self.polar_encoder(x_padded)

        feat_polar_packed = polar_vol_encoded.view(B, Zp, self.out_channels, Tp, Rp).permute(0, 2, 1, 3, 4).reshape(B,
                                                                                                                    self.out_channels * Zp,
                                                                                                                    Tp,
                                                                                                                    Rp)

        xs = self.bev_x_lin.to(device)
        ys = self.bev_y_lin.to(device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')

        rr_bev = torch.sqrt(xx ** 2 + yy ** 2 + 1e-6)
        tt_bev = torch.atan2(yy, xx)
        norm_theta = tt_bev / math.pi

        norm_rho = torch.zeros_like(rr_bev)
        mask_core = rr_bev < self.R_core
        norm_rho[mask_core] = rr_bev[mask_core] * self.alpha / self.R_core
        denom = max(self.R_max - self.R_core, 1e-3)
        norm_rho[~mask_core] = self.alpha + (rr_bev[~mask_core] - self.R_core) * (1 - self.alpha) / denom
        norm_rho = norm_rho * 2.0 - 1.0

        grid_query = torch.stack([norm_rho, norm_theta], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        grid_query_low = grid_query.to(feat_dtype)
        bev_out_packed = F.grid_sample(feat_polar_packed, grid_query_low, align_corners=self.align_corners,
                                       padding_mode='zeros')

        bev_out = self.z_compression(bev_out_packed)

        return bev_out

    def forward(self, img, points, camera2ego, lidar2ego, lidar2camera, lidar2image,
                camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix,
                img_metas=None,
                **kwargs):

        B, N, C_img, H_img, W_img = img.shape

        if self.training and self.use_checkpoint and img.requires_grad:
            cam_feats_2d = cp.checkpoint(self.get_cam_feats, img, use_reentrant=False)
        else:
            cam_feats_2d = self.get_cam_feats(img)

        cam_feats_2d = cam_feats_2d.view(B, N, -1, cam_feats_2d.shape[-2], cam_feats_2d.shape[-1])

        if self.training and self.use_checkpoint and cam_feats_2d.requires_grad:
            bev_out = cp.checkpoint(
                self._inner_view_transform,
                cam_feats_2d,
                lidar_aug_matrix,
                camera2ego,
                camera_intrinsics,
                img_aug_matrix,
                use_reentrant=False
            )
        else:
            bev_out = self._inner_view_transform(
                cam_feats_2d,
                lidar_aug_matrix,
                camera2ego,
                camera_intrinsics,
                img_aug_matrix
            )

        out = self.downsample_layer(bev_out)

        return out
