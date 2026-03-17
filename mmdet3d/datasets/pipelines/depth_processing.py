import numpy as np
import torch
import torch.nn.functional as F
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PointToMultiViewDepth(object):
    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width, lidar2image):
        height, width = height // self.downsample, width // self.downsample
        depth_map = np.zeros((height, width), dtype=np.float32)
        coor = self.project_points(points, lidar2image)
        coor[:, :2] /= self.downsample
        coor_x = np.round(coor[:, 0]).astype(np.int32)
        coor_y = np.round(coor[:, 1]).astype(np.int32)
        depth = coor[:, 2]
        valid_mask = (coor_x >= 0) & (coor_x < width) & \
                     (coor_y >= 0) & (coor_y < height) & \
                     (depth > self.grid_config[0]) & \
                     (depth < self.grid_config[1])
        coor_x = coor_x[valid_mask]
        coor_y = coor_y[valid_mask]
        depth = depth[valid_mask]
        sort_idx = np.argsort(-depth)
        coor_x = coor_x[sort_idx]
        coor_y = coor_y[sort_idx]
        depth = depth[sort_idx]
        depth_map[coor_y, coor_x] = depth
        d_tensor = torch.from_numpy(depth_map).unsqueeze(0).unsqueeze(0)
        k_size = 3
        padding = k_size // 2
        d_dilated = F.max_pool2d(d_tensor, kernel_size=k_size, stride=1, padding=padding)
        depth_map = d_dilated.squeeze().numpy()
        return depth_map

    def project_points(self, points, trans_mat):
        n_points = points.shape[0]
        points_hom = np.concatenate([points[:, :3], np.ones((n_points, 1))], axis=1)
        points_img = points_hom @ trans_mat.T
        points_img = points_img[:, :3]
        points_img[:, :2] /= (points_img[:, 2:3] + 1e-6)
        return points_img

    def __call__(self, results):
        points = results['points'].tensor.numpy()
        img_shape = results['img_shape']
        if isinstance(img_shape, list):
            img_h, img_w = img_shape[0][:2]
        elif isinstance(img_shape, tuple):
            if len(img_shape) == 3:
                img_h, img_w = img_shape[:2]
            elif len(img_shape) == 4:
                img_h, img_w = img_shape[1:3]
            else:
                img_h, img_w = img_shape[:2]
        else:
            img_h, img_w = img_shape[:2]
        lidar2camera = results['lidar2camera']
        camera_intrinsics = results['camera_intrinsics']
        img_aug_matrix = results['img_aug_matrix']

        gt_depths = []
        for i in range(len(lidar2camera)):
            l2c = lidar2camera[i]
            intrin = camera_intrinsics[i]
            aug = img_aug_matrix[i]

            view_pad = np.eye(4)
            view_pad[:3, :3] = intrin[:3, :3]
            view_pad[:3, 3] = intrin[:3, 3]

            aug_pad = np.eye(4)
            aug_pad[:3, :3] = aug[:3, :3]
            aug_pad[:3, 3] = aug[:3, 3]

            lidar2img_aug = aug_pad @ view_pad @ l2c
            depth_map = self.points2depthmap(points, img_h, img_w, lidar2img_aug)
            gt_depths.append(depth_map)

        results['gt_depth'] = np.stack(gt_depths)
        return results
