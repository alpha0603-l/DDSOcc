import torch
from torch.autograd import Function
from torch.cuda.amp import custom_fwd

try:
    from . import knn_ext
except ImportError:
    knn_ext = None


class KNN(Function):
    r"""KNN (CUDA) based on heap data structure.
    Modified from `PAConv <https://github.com/CVMI-Lab/PAConv/tree/main/
    scene_seg/lib/pointops/src/knnquery_heap>`_.

    Find k-nearest points.
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # 强制float32，因为核函数不支持FP16
    def forward(
            ctx, k: int, xyz: torch.Tensor, center_xyz: torch.Tensor = None, transposed: bool = False
    ) -> torch.Tensor:
        """Forward.

        Args:
            k (int): number of nearest neighbors.
            xyz (Tensor): (B, N, 3) if transposed == False, else (B, 3, N).
                xyz coordinates of the features.
            center_xyz (Tensor): (B, npoint, 3) if transposed == False,
                else (B, 3, npoint). centers of the knn query.
            transposed (bool): whether the input tensors are transposed.
                defaults to False. Should not expicitly use this keyword
                when calling knn (=KNN.apply), just add the fourth param.

        Returns:
            Tensor: (B, k, npoint) tensor with the indicies of
                the features that form k-nearest neighbours.
        """
        assert k > 0
        if knn_ext is None:
            raise ImportError("knn_ext has not been compiled.")

        if center_xyz is None:
            center_xyz = xyz

        if transposed:
            xyz = xyz.transpose(2, 1).contiguous()
            center_xyz = center_xyz.transpose(2, 1).contiguous()

        xyz = xyz.contiguous()
        center_xyz = center_xyz.contiguous()

        center_xyz_device = center_xyz.device
        assert (
                center_xyz_device == xyz.device
        ), "center_xyz and xyz should be put on the same device"

        B, npoint, _ = center_xyz.shape
        N = xyz.shape[1]

        idx = torch.zeros((B, npoint, k), dtype=torch.int32, device=center_xyz_device)
        dist2 = torch.zeros((B, npoint, k), dtype=torch.float32, device=center_xyz_device)

        knn_ext.knn_wrapper(B, N, npoint, k, xyz, center_xyz, idx, dist2)

        idx = idx.transpose(2, 1).contiguous()
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None


knn = KNN.apply
