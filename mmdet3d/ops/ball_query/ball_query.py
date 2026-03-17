import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
try:
    from . import ball_query_ext
except ImportError:
    ball_query_ext = None


class BallQuery(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
            ctx,
            min_radius: float,
            max_radius: float,
            sample_num: int,
            xyz: torch.Tensor,
            center_xyz: torch.Tensor,
    ) -> torch.Tensor:
        """forward.

        Args:
            min_radius (float): minimum radius of the balls.
            max_radius (float): maximum radius of the balls.
            sample_num (int): maximum number of features in the balls.
            xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            center_xyz (Tensor): (B, npoint, 3) centers of the ball query.

        Returns:
            Tensor: (B, npoint, nsample) tensor with the indicies of
                the features that form the query balls.
        """
        if ball_query_ext is None:
            raise ImportError(
                "ball_query_ext has not been compiled or loaded. "
                "Please check your installation."
            )

        xyz = xyz.contiguous()
        center_xyz = center_xyz.contiguous()

        assert min_radius < max_radius

        B, N, _ = xyz.size()
        npoint = center_xyz.size(1)

        idx = torch.zeros(
            (B, npoint, sample_num),
            dtype=torch.int32,
            device=xyz.device
        )

        ball_query_ext.ball_query_wrapper(
            B, N, npoint, min_radius, max_radius, sample_num, center_xyz, xyz, idx
        )

        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply
