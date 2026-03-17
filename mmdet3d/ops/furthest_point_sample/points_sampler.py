import torch
def force_fp32(apply_to=None, out_fp16=False):
    def decorator(func):
        return func
    return decorator

from torch import nn as nn
from typing import List

from .furthest_point_sample import furthest_point_sample, furthest_point_sample_with_dist
from .utils import calc_square_dist


def get_sampler_type(sampler_type):
    if sampler_type == "D-FPS":
        sampler = DFPS_Sampler
    elif sampler_type == "F-FPS":
        sampler = FFPS_Sampler
    elif sampler_type == "FS":
        sampler = FS_Sampler
    else:
        raise ValueError(
            'Only "sampler_type" of "D-FPS", "F-FPS", or "FS"' f" are supported, got {sampler_type}"
        )
    return sampler


class Points_Sampler(nn.Module):
    def __init__(
            self,
            num_point: List[int],
            fps_mod_list: List[str] = ["D-FPS"],
            fps_sample_range_list: List[int] = [-1],
    ):
        super(Points_Sampler, self).__init__()
        assert len(num_point) == len(fps_mod_list) == len(fps_sample_range_list)
        self.num_point = num_point
        self.fps_sample_range_list = fps_sample_range_list
        self.samplers = nn.ModuleList()
        for fps_mod in fps_mod_list:
            self.samplers.append(get_sampler_type(fps_mod)())
        self.fp16_enabled = False

    @force_fp32()
    def forward(self, points_xyz, features):
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates.
            features (Tensor): (B, C, N) Descriptors.
        """
        indices = []
        last_fps_end_index = 0

        for fps_sample_range, sampler, npoint in zip(
                self.fps_sample_range_list, self.samplers, self.num_point
        ):
            if fps_sample_range >= points_xyz.shape[1]:
                fps_sample_range = -1

            if fps_sample_range == -1:
                sample_points_xyz = points_xyz[:, last_fps_end_index:]
                sample_features = (
                    features[:, :, last_fps_end_index:] if features is not None else None
                )
            else:
                sample_points_xyz = points_xyz[:, last_fps_end_index:fps_sample_range]
                sample_features = (
                    features[:, :, last_fps_end_index:fps_sample_range]
                    if features is not None
                    else None
                )

            fps_idx = sampler(sample_points_xyz.contiguous(), sample_features, npoint)

            indices.append(fps_idx + last_fps_end_index)

            if fps_sample_range != -1:
                last_fps_end_index = fps_sample_range
            else:
                last_fps_end_index += sample_points_xyz.shape[1]

        indices = torch.cat(indices, dim=1)
        return indices

class DFPS_Sampler(nn.Module):
    def __init__(self):
        super(DFPS_Sampler, self).__init__()

    def forward(self, points, features, npoint):
        return furthest_point_sample(points.contiguous(), npoint)

class FFPS_Sampler(nn.Module):
    def __init__(self):
        super(FFPS_Sampler, self).__init__()

    def forward(self, points, features, npoint):
        assert features is not None, "feature input to FFPS_Sampler should not be None"
        features_for_fps = torch.cat([points, features.transpose(1, 2)], dim=2)
        features_dist = calc_square_dist(features_for_fps, features_for_fps, norm=False)
        return furthest_point_sample_with_dist(features_dist, npoint)


class FS_Sampler(nn.Module):
    def __init__(self):
        super(FS_Sampler, self).__init__()

    def forward(self, points, features, npoint):
        assert features is not None, "feature input to FS_Sampler should not be None"
        features_for_fps = torch.cat([points, features.transpose(1, 2)], dim=2)
        features_dist = calc_square_dist(features_for_fps, features_for_fps, norm=False)
        fps_idx_ffps = furthest_point_sample_with_dist(features_dist, npoint)
        fps_idx_dfps = furthest_point_sample(points, npoint)
        return torch.cat([fps_idx_ffps, fps_idx_dfps], dim=1)
