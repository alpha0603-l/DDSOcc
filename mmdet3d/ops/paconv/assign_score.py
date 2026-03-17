import torch
from torch.autograd import Function

try:
    from . import assign_score_withk_ext
except ImportError:
    assign_score_withk_ext = None

class AssignScoreWithK(Function):
    @staticmethod
    def forward(ctx, scores, point_features, center_features, knn_idx, aggregate="sum"):
        if assign_score_withk_ext is None:
            raise ImportError("assign_score_withk_ext not compiled! Check setup.py")

        agg = {"sum": 0, "avg": 1, "max": 2}

        B, N, M, out_dim = point_features.size()
        _, npoint, K, _ = scores.size()

        output = point_features.new_zeros((B, out_dim, npoint, K))
        assign_score_withk_ext.assign_score_withk_forward_wrapper(
            B,
            N,
            npoint,
            M,
            K,
            out_dim,
            agg[aggregate],
            point_features.contiguous(),
            center_features.contiguous(),
            scores.contiguous(),
            knn_idx.contiguous(),
            output,
        )

        ctx.save_for_backward(output, point_features, center_features, scores, knn_idx)
        ctx.agg = agg[aggregate]

        return output

    @staticmethod
    def backward(ctx, grad_out):
        _, point_features, center_features, scores, knn_idx = ctx.saved_tensors
        agg = ctx.agg

        B, N, M, out_dim = point_features.size()
        _, npoint, K, _ = scores.size()

        grad_point_features = point_features.new_zeros(point_features.shape)
        grad_center_features = center_features.new_zeros(center_features.shape)
        grad_scores = scores.new_zeros(scores.shape)

        assign_score_withk_ext.assign_score_withk_backward_wrapper(
            B,
            N,
            npoint,
            M,
            K,
            out_dim,
            agg,
            grad_out.contiguous(),
            point_features.contiguous(),
            center_features.contiguous(),
            scores.contiguous(),
            knn_idx.contiguous(),
            grad_point_features,
            grad_center_features,
            grad_scores,
        )

        return grad_scores, grad_point_features, grad_center_features, None, None

assign_score_withk = AssignScoreWithK.apply
