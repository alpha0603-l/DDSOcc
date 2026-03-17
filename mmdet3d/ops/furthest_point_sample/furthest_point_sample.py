import torch
from torch.autograd import Function
from torch.cuda.amp import custom_fwd

try:
    from . import furthest_point_sample_ext
except ImportError:
    furthest_point_sample_ext = None


class FurthestPointSampling(Function):
    """Furthest Point Sampling.

    Uses iterative furthest point sampling to select a set of features whose
    corresponding points have the furthest distance.
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # FPS 距离计算必须用 float32
    def forward(ctx, points_xyz: torch.Tensor, num_points: int) -> torch.Tensor:
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) where N > num_points.
            num_points (int): Number of points in the sampled set.

        Returns:
             Tensor: (B, num_points) indices of the sampled points.
        """
        if furthest_point_sample_ext is None:
            raise ImportError("furthest_point_sample_ext not compiled")

        points_xyz = points_xyz.contiguous()

        B, N = points_xyz.size()[:2]

        # 现代写法
        output = torch.zeros((B, num_points), dtype=torch.int32, device=points_xyz.device)
        temp = torch.full((B, N), 1e10, dtype=torch.float32, device=points_xyz.device)

        furthest_point_sample_ext.furthest_point_sampling_wrapper(
            B, N, num_points, points_xyz, temp, output
        )
        ctx.mark_non_differentiable(output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


class FurthestPointSamplingWithDist(Function):
    """Furthest Point Sampling With Distance.

    Uses iterative furthest point sampling to select a set of features whose
    corresponding points have the furthest distance.
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, points_dist: torch.Tensor, num_points: int) -> torch.Tensor:
        """forward.

        Args:
            points_dist (Tensor): (B, N, N) Distance between each point pair.
            num_points (int): Number of points in the sampled set.

        Returns:
             Tensor: (B, num_points) indices of the sampled points.
        """
        if furthest_point_sample_ext is None:
            raise ImportError("furthest_point_sample_ext not compiled")

        points_dist = points_dist.contiguous()

        B, N, _ = points_dist.size()

        # 现代写法
        output = torch.zeros((B, num_points), dtype=torch.int32, device=points_dist.device)
        temp = torch.full((B, N), 1e10, dtype=torch.float32, device=points_dist.device)

        furthest_point_sample_ext.furthest_point_sampling_with_dist_wrapper(
            B, N, num_points, points_dist, temp, output
        )
        ctx.mark_non_differentiable(output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply
furthest_point_sample_with_dist = FurthestPointSamplingWithDist.apply
