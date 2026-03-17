import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    from . import gather_points_ext
except ImportError:
    gather_points_ext = None


class GatherPoints(Function):
    """Gather Points.

    Gather points with given index.
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # 强制 float32
    def forward(ctx, features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            features (Tensor): (B, C, N) features to gather.
            indices (Tensor): (B, M) where M is the number of points.

        Returns:
            Tensor: (B, C, M) where M is the number of points.
        """
        if gather_points_ext is None:
            raise ImportError("gather_points_ext has not been compiled.")

        features = features.contiguous()
        indices = indices.contiguous()

        B, npoint = indices.size()
        _, C, N = features.size()

        output = torch.zeros(
            (B, C, npoint),
            dtype=features.dtype,
            device=features.device
        )

        gather_points_ext.gather_points_wrapper(
            B, C, N, npoint, features, indices, output
        )

        ctx.for_backwards = (indices, C, N)
        ctx.mark_non_differentiable(indices)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = torch.zeros(
            (B, C, N),
            dtype=grad_out.dtype,
            device=grad_out.device
        )

        grad_out_data = grad_out.contiguous()

        gather_points_ext.gather_points_grad_wrapper(
            B, C, N, npoint, grad_out_data, idx, grad_features
        )
        return grad_features, None


gather_points = GatherPoints.apply
