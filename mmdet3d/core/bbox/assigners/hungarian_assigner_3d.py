import torch

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None
try:
    from mmdet.registry import TASK_UTILS
except (ImportError, ModuleNotFoundError):
    try:
        from mmengine.registry import TASK_UTILS
    except (ImportError, ModuleNotFoundError):
        class MockRegistry:
            def register_module(self, name=None, force=False, module=None):
                if module is not None: return module

                def _register(cls): return cls

                return _register

            def build(self, cfg, *args, **kwargs): return None


        TASK_UTILS = MockRegistry()

try:
    from mmdet.models.task_modules.assigners import AssignResult, BaseAssigner
except (ImportError, ModuleNotFoundError):
    try:
        from mmdet.core.bbox.assigners import AssignResult, BaseAssigner
    except (ImportError, ModuleNotFoundError):
        class AssignResult:
            def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
                self.num_gts, self.gt_inds = num_gts, gt_inds
                self.max_overlaps, self.labels = max_overlaps, labels

            @property
            def num_preds(self): return len(self.gt_inds)


        class BaseAssigner:
            def assign(self, *args, **kwargs): pass


def build_match_cost(cfg):
    return TASK_UTILS.build(cfg)


def normalize_bbox(bboxes, pc_range):
    if pc_range is None: return bboxes
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()
    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vel_x = bboxes[..., 7:8]
        vel_y = bboxes[..., 8:9]
        return torch.cat([cx, cy, cz, w, l, h, rot, vel_x, vel_y], dim=-1)
    return torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)


@TASK_UTILS.register_module(force=True)
class HungarianAssigner3D(BaseAssigner):


    def __init__(
            self,
            cls_cost=dict(type="ClassificationCost", weight=1.0),
            reg_cost=dict(type="BBoxL1Cost", weight=1.0),
            iou_cost=dict(type="IoUCost", weight=0.0),
            pc_range=None,
    ):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.pc_range = pc_range

    def assign(
            self, bbox_pred, cls_pred, gt_bboxes, gt_labels, gt_bboxes_ignore=None, eps=1e-7
    ):
        assert (
                gt_bboxes_ignore is None
        ), "Only case when gt_bboxes_ignore is None is supported."
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        assigned_gt_inds = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

        cls_cost = self.cls_cost(cls_pred, gt_labels)

        normalized_gt_bboxes = normalize_bbox(gt_bboxes, self.pc_range)
        slice_dim = min(bbox_pred.size(1), 8)
        reg_cost = self.reg_cost(bbox_pred[:, :slice_dim], normalized_gt_bboxes[:, :slice_dim])


        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('scipy required')

        cost_np = cost.numpy() if hasattr(cost, 'numpy') else cost
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost_np)

        matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred.device)

        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
