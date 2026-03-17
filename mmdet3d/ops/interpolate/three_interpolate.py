import torch
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
from typing import Tuple

try:
    from . import interpolate_ext
except ImportError:
    interpolate_ext = None


class ThreeInterpolate(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
            ctx, features: torch.Tensor, indices: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """Performs weighted linear interpolation on 3 features.

        Args:
            features (Tensor): (B, C, M) Features descriptors to be
                interpolated from
            indices (Tensor): (B, n, 3) index three nearest neighbors
                of the target features in features
            weight (Tensor): (B, n, 3) weights of interpolation

        Returns:
            Tensor: (B, C, N) tensor of the interpolated features
        """
        if interpolate_ext is None:
            raise ImportError("interpolate_ext not compiled")

        features = features.contiguous()
        indices = indices.contiguous()
        weight = weight.contiguous()

        B, c, m = features.size()
        n = indices.size(1)
        ctx.three_interpolate_for_backward = (indices, weight, m)

        output = torch.zeros((B, c, n), dtype=torch.float32, device=features.device)

        interpolate_ext.three_interpolate_wrapper(B, c, m, n, features, indices, weight, output)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward of three interpolate.

        Args:
            grad_out (Tensor): (B, C, N) tensor with gradients of outputs

        Returns:
            Tensor: (B, C, M) tensor with gradients of features
        """
        idx, weight, m = ctx.three_interpolate_for_backward

        # 梯度可能是半精度的，需要转为 float32
        grad_out = grad_out.contiguous().float()

        B, c, n = grad_out.size()

        grad_features = torch.zeros((B, c, m), dtype=torch.float32, device=grad_out.device)

        interpolate_ext.three_interpolate_grad_wrapper(
            B, c, n, m, grad_out, idx, weight, grad_features
        )
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply
