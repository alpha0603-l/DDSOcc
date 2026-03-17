import torch
from mmdet.registry import TASK_UTILS

@TASK_UTILS.register_module()
class BBox3DL1Cost:

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        bbox_pred = bbox_pred.float()
        gt_bboxes = gt_bboxes.float()

        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight
