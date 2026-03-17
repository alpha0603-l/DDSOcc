import torch
try:
    from mmdet.structures.bbox import bbox_overlaps
except ImportError:
    from mmdet.models.task_modules.assigners.iou_calculators import bbox_overlaps
from mmdet.registry import TASK_UTILS
from ..structures import get_box_type
try:
    from mmdet3d.ops.iou3d import iou3d_cuda
except ImportError:
    pass

@TASK_UTILS.register_module()
class BboxOverlapsNearest3D:

    def __init__(self, coordinate="lidar"):
        assert coordinate in ["camera", "lidar", "depth"]
        self.coordinate = coordinate

    def __call__(self, bboxes1, bboxes2, mode="iou", is_aligned=False):
        """Calculate nearest 3D IoU."""
        return bbox_overlaps_nearest_3d(
            bboxes1, bboxes2, mode, is_aligned, self.coordinate
        )

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(coordinate={self.coordinate}"
        return repr_str


@TASK_UTILS.register_module()  # [修复] 注册到 TASK_UTILS
class BboxOverlaps3D:
    """3D IoU Calculator.

    Args:
        coordinate (str): The coordinate system, valid options are
            'camera', 'lidar', and 'depth'.
    """

    def __init__(self, coordinate):
        assert coordinate in ["camera", "lidar", "depth"]
        self.coordinate = coordinate

    def __call__(self, bboxes1, bboxes2, mode="iou"):
        """Calculate 3D IoU using cuda implementation."""
        return bbox_overlaps_3d(bboxes1, bboxes2, mode, self.coordinate)

    def __repr__(self):
        """str: return a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f"(coordinate={self.coordinate}"
        return repr_str


def bbox_overlaps_nearest_3d(
    bboxes1, bboxes2, mode="iou", is_aligned=False, coordinate="lidar"
):
    """Calculate nearest 3D IoU."""
    assert bboxes1.size(-1) == bboxes2.size(-1) >= 7

    box_type, _ = get_box_type(coordinate)

    bboxes1 = box_type(bboxes1, box_dim=bboxes1.shape[-1])
    bboxes2 = box_type(bboxes2, box_dim=bboxes2.shape[-1])
    bboxes1_bev = bboxes1.nearest_bev
    bboxes2_bev = bboxes2.nearest_bev

    ret = bbox_overlaps(bboxes1_bev, bboxes2_bev, mode=mode, is_aligned=is_aligned)
    return ret


def bbox_overlaps_3d(bboxes1, bboxes2, mode="iou", coordinate="camera"):
    """Calculate 3D IoU using cuda implementation."""
    assert bboxes1.size(-1) == bboxes2.size(-1) >= 7

    box_type, _ = get_box_type(coordinate)

    bboxes1 = box_type(bboxes1, box_dim=bboxes1.shape[-1])
    bboxes2 = box_type(bboxes2, box_dim=bboxes2.shape[-1])

    return bboxes1.overlaps(bboxes1, bboxes2, mode=mode)


@TASK_UTILS.register_module()
class AxisAlignedBboxOverlaps3D:
    """Axis-aligned 3D Overlaps (IoU) Calculator."""

    def __call__(self, bboxes1, bboxes2, mode="iou", is_aligned=False):
        """Calculate IoU between 2D bboxes."""
        assert bboxes1.size(-1) == bboxes2.size(-1) == 6
        return axis_aligned_bbox_overlaps_3d(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + "()"
        return repr_str


def axis_aligned_bbox_overlaps_3d(
    bboxes1, bboxes2, mode="iou", is_aligned=False, eps=1e-6
):
    """Calculate overlap between two set of axis aligned 3D bboxes."""

    assert mode in ["iou", "giou"], f"Unsupported mode {mode}"
    # Either the boxes are empty or the length of boxes's last dimenstion is 6
    assert bboxes1.size(-1) == 6 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 6 or bboxes2.size(0) == 0

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (
        (bboxes1[..., 3] - bboxes1[..., 0])
        * (bboxes1[..., 4] - bboxes1[..., 1])
        * (bboxes1[..., 5] - bboxes1[..., 2])
    )
    area2 = (
        (bboxes2[..., 3] - bboxes2[..., 0])
        * (bboxes2[..., 4] - bboxes2[..., 1])
        * (bboxes2[..., 5] - bboxes2[..., 2])
    )

    if is_aligned:
        lt = torch.max(bboxes1[..., :3], bboxes2[..., :3])  # [B, rows, 3]
        rb = torch.min(bboxes1[..., 3:], bboxes2[..., 3:])  # [B, rows, 3]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

        if mode in ["iou", "giou"]:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :3], bboxes2[..., :3])
            enclosed_rb = torch.max(bboxes1[..., 3:], bboxes2[..., 3:])
    else:
        lt = torch.max(
            bboxes1[..., :, None, :3], bboxes2[..., None, :, :3]
        )  # [B, rows, cols, 3]
        rb = torch.min(
            bboxes1[..., :, None, 3:], bboxes2[..., None, :, 3:]
        )  # [B, rows, cols, 3]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 3]
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]

        if mode in ["iou", "giou"]:
            union = area1[..., None] + area2[..., None, :] - overlap
        if mode == "giou":
            enclosed_lt = torch.min(
                bboxes1[..., :, None, :3], bboxes2[..., None, :, :3]
            )
            enclosed_rb = torch.max(
                bboxes1[..., :, None, 3:], bboxes2[..., None, :, 3:]
            )

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ["iou"]:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1] * enclose_wh[..., 2]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious
