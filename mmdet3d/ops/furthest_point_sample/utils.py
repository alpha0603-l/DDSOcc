import torch

def calc_square_dist(point_feat_a, point_feat_b, norm=True):
    """Calculating square distance between a and b.

    Args:
        point_feat_a (Tensor): (B, N, C) Feature vector of each point.
        point_feat_b (Tensor): (B, M, C) Feature vector of each point.
        norm (Bool): Whether to normalize the distance.
            Default: True.

    Returns:
        Tensor: (B, N, M) Distance between each pair points.
    """
    length_a = point_feat_a.shape[1]
    length_b = point_feat_b.shape[1]
    num_channel = point_feat_a.shape[-1]

    a_square = torch.sum(point_feat_a.pow(2), dim=-1, keepdim=True)  # [B, N, 1]
    b_square = torch.sum(point_feat_b.pow(2), dim=-1, keepdim=True)  # [B, M, 1] -> 转置后 [B, 1, M]

    coor = torch.matmul(point_feat_a, point_feat_b.transpose(1, 2))

    dist = a_square + b_square.transpose(1, 2) - 2 * coor

    dist = torch.relu(dist)

    if norm:
        dist = torch.sqrt(dist) / num_channel
    return dist
