from typing import Tuple

import torch
from torch import nn
import numpy as np
from scipy.spatial.transform import Rotation as R_tool

from mmengine.model import BaseModule

from mmdet3d.ops import bev_pool

__all__ = ["BaseTransform", "BaseDepthTransform"]


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx


class BaseTransform(BaseModule):
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
            init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False

    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = (
            torch.arange(*self.dbound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        frustum = torch.stack((xs, ys, ds), -1)  # torch.Size([118, 32, 88, 3])
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(
            self,
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            **kwargs,
    ):
        if camera2lidar_rots.dtype != torch.float32: camera2lidar_rots = camera2lidar_rots.float()
        if camera2lidar_trans.dtype != torch.float32: camera2lidar_trans = camera2lidar_trans.float()
        if intrins.dtype != torch.float32: intrins = intrins.float()
        if post_rots.dtype != torch.float32: post_rots = post_rots.float()
        if post_trans.dtype != torch.float32: post_trans = post_trans.float()

        B, N, _ = camera2lidar_trans.shape

        points = self.frustum.to(post_trans.device) - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            if extra_rots.dtype != torch.float32: extra_rots = extra_rots.float()
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            if extra_trans.dtype != torch.float32: extra_trans = extra_trans.float()
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        return points

    def get_cam_feats(self, x):
        raise NotImplementedError

    def bev_pool(self, geom_feats, x):
        if x.dtype != torch.float32: x = x.float()
        if geom_feats.dtype != torch.float32: geom_feats = geom_feats.float()

        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        x = x.reshape(Nprime, C)

        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        kept = (
                (geom_feats[:, 0] >= 0)
                & (geom_feats[:, 0] < self.nx[0])
                & (geom_feats[:, 1] >= 0)
                & (geom_feats[:, 1] < self.nx[1])
                & (geom_feats[:, 2] >= 0)
                & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]
        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])
        final = torch.cat(x.unbind(dim=2), 1)
        return final

    def forward(
            self,
            img,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            **kwargs,
    ):
        NOISE_ROT_DEG = 1.0
        NOISE_TRANS_M = 0.1

        def _inject(mats):
            B, N = mats.shape[:2]
            mats_view = mats.view(-1, 4, 4)
            new_mats = []
            for i in range(mats_view.shape[0]):
                m_np = mats_view[i].detach().cpu().numpy()
                r_off = np.random.normal(0, NOISE_ROT_DEG, 3)
                rot_m = R_tool.from_euler('xyz', r_off, degrees=True).as_matrix()
                t_off = np.random.normal(0, NOISE_TRANS_M, 3)
                noise_m = np.eye(4); noise_m[:3, :3] = rot_m; noise_m[:3, 3] = t_off
                new_mats.append(torch.from_numpy(m_np @ noise_m).to(mats.device, mats.dtype))
            return torch.stack(new_mats).view(B, N, 4, 4)
        camera2lidar = _inject(camera2lidar)

        rots = camera2ego[..., :3, :3]
        trans = camera2ego[..., :3, 3]
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img)
        x = self.bev_pool(geom, x)
        return x


class BaseDepthTransform(BaseTransform):
    def forward(
            self,
            img,
            points,
            sensor2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            cam_intrinsic,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            metas,
            **kwargs,
    ):
        NOISE_ROT_DEG = 1.0
        NOISE_TRANS_M = 0.1

        def _inject(mats):
            B, N = mats.shape[:2]
            mats_view = mats.view(-1, 4, 4)
            new_mats = []
            for i in range(mats_view.shape[0]):
                m_np = mats_view[i].detach().cpu().numpy()
                r_off = np.random.normal(0, NOISE_ROT_DEG, 3)
                rot_m = R_tool.from_euler('xyz', r_off, degrees=True).as_matrix()
                t_off = np.random.normal(0, NOISE_TRANS_M, 3)
                noise_m = np.eye(4); noise_m[:3, :3] = rot_m; noise_m[:3, 3] = t_off
                new_mats.append(torch.from_numpy(m_np @ noise_m).to(mats.device, mats.dtype))
            return torch.stack(new_mats).view(B, N, 4, 4)

        camera2lidar = _inject(camera2lidar)

        rots = sensor2ego[..., :3, :3]
        trans = sensor2ego[..., :3, 3]
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        batch_size = len(points)
        depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(
            points[0].device
        )

        for b in range(batch_size):
            cur_coords = points[b][:, :3].float()
            cur_img_aug_matrix = img_aug_matrix[b].float()
            cur_lidar_aug_matrix = lidar_aug_matrix[b].float()
            cur_lidar2image = lidar2image[b].float()

            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)
            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                    (cur_coords[..., 0] < self.image_size[0])
                    & (cur_coords[..., 0] >= 0)
                    & (cur_coords[..., 1] < self.image_size[1])
                    & (cur_coords[..., 1] >= 0)
            )
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img, depth)
        x = self.bev_pool(geom, x)
        return x