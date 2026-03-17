from mmdet.registry import TASK_UTILS
try:
    from mmdet.models.task_modules.assigners.match_costs import (
        BBoxL1Cost, ClassificationCost, IoUCost, FocalLossCost, DiceCost
    )
except ImportError:
    class BBoxL1Cost: pass
    class ClassificationCost: pass
    class IoUCost: pass
    class FocalLossCost: pass
    class DiceCost: pass
from .match_cost import BBox3DL1Cost

__all__ = [
    'BBox3DL1Cost',
    'BBoxL1Cost', 'ClassificationCost', 'IoUCost', 'FocalLossCost', 'DiceCost'
]
