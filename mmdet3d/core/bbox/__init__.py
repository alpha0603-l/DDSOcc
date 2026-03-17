from mmdet.registry import TASK_UTILS
from mmdet.models.task_modules import (
    AssignResult,
    BaseAssigner,
    MaxIoUAssigner,
    BaseSampler,
    PseudoSampler,
    RandomSampler,
    SamplingResult,
)
def build_assigner(cfg, default_args=None):
    return TASK_UTILS.build(cfg, default_args=default_args)

def build_sampler(cfg, default_args=None):
    return TASK_UTILS.build(cfg, default_args=default_args)

def build_bbox_coder(cfg, default_args=None):
    return TASK_UTILS.build(cfg, default_args=default_args)

def build_match_cost(cfg, default_args=None):
    return TASK_UTILS.build(cfg, default_args=default_args)

# Assigners
from .assigners import HungarianAssigner3D, HeuristicAssigner3D

# Match Costs
from .match_costs import BBox3DL1Cost, BBoxL1Cost, ClassificationCost, IoUCost

# Coders
from .coders import DeltaXYZWLHRBBoxCoder, CenterPointBBoxCoder

# Structures
try:
    from .structures import (BaseInstance3DBoxes, Box3DMode, CameraInstance3DBoxes,
                             Coord3DMode, DepthInstance3DBoxes,
                             LiDARInstance3DBoxes, get_box_type, limit_period,
                             mono_cam_box2vis, points_cam2img, xywhr2xyxyr)
except ImportError:
    from mmdet3d.structures import (BaseInstance3DBoxes, Box3DMode, CameraInstance3DBoxes,
                                    Coord3DMode, DepthInstance3DBoxes,
                                    LiDARInstance3DBoxes, get_box_type, limit_period,
                                    mono_cam_box2vis, points_cam2img, xywhr2xyxyr)

from .iou_calculators import (AxisAlignedBboxOverlaps3D, BboxOverlaps3D,
                              BboxOverlapsNearest3D,
                              axis_aligned_bbox_overlaps_3d, bbox_overlaps_3d,
                              bbox_overlaps_nearest_3d)

from .samplers import (CombinedSampler,
                       InstanceBalancedPosSampler, IoUBalancedNegSampler)


__all__ = [
    'AssignResult', 'BaseAssigner', 'MaxIoUAssigner', 'BaseSampler',
    'PseudoSampler', 'RandomSampler', 'SamplingResult',
    'build_assigner', 'build_sampler', 'build_bbox_coder', 'build_match_cost',
    'HungarianAssigner3D', 'HeuristicAssigner3D',
    'DeltaXYZWLHRBBoxCoder', 'CenterPointBBoxCoder',
    'BBox3DL1Cost', 'BBoxL1Cost', 'ClassificationCost', 'IoUCost',
    'BaseInstance3DBoxes', 'Box3DMode', 'CameraInstance3DBoxes',
    'Coord3DMode', 'DepthInstance3DBoxes',
    'LiDARInstance3DBoxes', 'get_box_type', 'limit_period',
    'mono_cam_box2vis', 'points_cam2img', 'xywhr2xyxyr',
    'AxisAlignedBboxOverlaps3D', 'BboxOverlaps3D', 'BboxOverlapsNearest3D',
    'axis_aligned_bbox_overlaps_3d', 'bbox_overlaps_3d', 'bbox_overlaps_nearest_3d',
    'CombinedSampler', 'InstanceBalancedPosSampler', 'IoUBalancedNegSampler'
]
