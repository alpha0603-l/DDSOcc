from .hungarian_assigner import HungarianAssigner3D, HeuristicAssigner3D
try:
    from .hungarian_assigner_3d import HungarianAssigner3D as HungarianAssigner3Dv2
except ImportError:
    pass

try:
    from mmdet.models.task_modules.assigners import AssignResult, BaseAssigner, MaxIoUAssigner
except (ImportError, ModuleNotFoundError):
    try:
        from mmdet.core.bbox.assigners import AssignResult, BaseAssigner, MaxIoUAssigner
    except (ImportError, ModuleNotFoundError):
        class AssignResult:
            def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
                self.num_gts, self.gt_inds, self.max_overlaps, self.labels = num_gts, gt_inds, max_overlaps, labels
            @property
            def num_preds(self): return len(self.gt_inds)
        class BaseAssigner:
            def assign(self, *args, **kwargs): pass
        class MaxIoUAssigner(BaseAssigner):
            pass

__all__ = [
    'AssignResult', 'BaseAssigner', 'MaxIoUAssigner',
    'HungarianAssigner3D', 'HeuristicAssigner3D'
]
