import torch
from torch.autograd import Function
from torch.cuda.amp import custom_fwd
from typing import Tuple

try:
    from . import interpolate_ext
except ImportError:
    interpolate_ext = None


class ThreeNN(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
            ctx, target: torch.Tensor, source: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find the top-3 nearest neighbors of the target set from the source
        set.

        Args:
            target (Tensor): shape (B, N, 3), points set that needs to
                find the nearest neighbors.
            source (Tensor): shape (B, M, 3), points set that is used
                to find the nearest neighbors of points in target set.

        Returns:
            Tensor: shape (B, N, 3), L2 distance of each point in target
                set to their corresponding nearest neighbors.
        """
        if interpolate_ext is None:
            raise ImportError("interpolate_ext not compiled")

        target = target.contiguous()
        source = source.contiguous()

        B, N, _ = target.size()
        m = source.size(1)

        dist2 = torch.zeros((B, N, 3), dtype=torch.float32, device=target.device)
        idx = torch.zeros((B, N, 3), dtype=torch.int32, device=target.device)

        interpolate_ext.three_nn_wrapper(B, N, m, target, source, dist2, idx)

        ctx.mark_non_differentiable(idx)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply
