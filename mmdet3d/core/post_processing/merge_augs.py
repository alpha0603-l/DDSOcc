import torch
def multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None):
    num_classes = multi_scores.size(1) - 1
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]
    valid_mask = scores > score_thr
    from torchvision.ops import nms

    dets, labels = [], []
    for i in range(num_classes):
        cls_inds = valid_mask[:, i]
        if not cls_inds.any():
            continue

        _bboxes = bboxes[cls_inds, i]
        _scores = scores[cls_inds, i]
        keep = nms(_bboxes, _scores, nms_cfg.get('iou_threshold', 0.5))
        dets.append(torch.cat([_bboxes[keep], _scores[keep][:, None]], dim=1))
        labels.append(torch.full((len(keep),), i, dtype=torch.long, device=_bboxes.device))

    if not dets:
        return torch.zeros((0, 5), device=multi_bboxes.device), \
            torch.zeros((0,), dtype=torch.long, device=multi_bboxes.device)

    dets = torch.cat(dets, dim=0)
    labels = torch.cat(labels, dim=0)

    if max_num > 0 and dets.shape[0] > max_num:
        _, inds = dets[:, 4].sort(descending=True)
        inds = inds[:max_num]
        dets = dets[inds]
        labels = labels[inds]

    return dets, labels

def merge_aug_proposals(aug_proposals, img_metas, cfg):
    recovered_proposals = []
    for proposals, _ in zip(aug_proposals, img_metas):
        recovered_proposals.append(proposals)
    return torch.cat(recovered_proposals, dim=0)


def merge_aug_bboxes(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg):
    recovered_bboxes = []
    for bboxes, _ in zip(aug_bboxes, img_metas):
        recovered_bboxes.append(bboxes)

    bboxes = torch.cat(recovered_bboxes, dim=0)
    scores = torch.cat(aug_scores, dim=0)

    if rcnn_test_cfg is None:
        return bboxes, scores

    det_bboxes, det_labels = multiclass_nms(
        bboxes,
        scores,
        rcnn_test_cfg.score_thr,
        rcnn_test_cfg.nms,
        rcnn_test_cfg.max_per_img
    )
    return det_bboxes, det_labels


def merge_aug_scores(aug_scores, img_metas, weights=None):
    if weights is None:
        return torch.mean(torch.stack(aug_scores), dim=0)
    else:
        weights = torch.tensor(weights).to(aug_scores[0].device)
        return torch.sum(torch.stack(aug_scores) * weights[:, None, None], dim=0)


def merge_aug_masks(aug_masks, img_metas, weights=None):
    return torch.mean(torch.stack(aug_masks), dim=0)
