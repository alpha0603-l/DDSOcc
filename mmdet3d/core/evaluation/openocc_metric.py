import torch
import numpy as np
import copy

import os.path as osp
import shutil
import tempfile
import torch.distributed as dist
try:
    from mmengine.dist import get_dist_info
    import mmengine
except ImportError:
    from mmcv.runner import get_dist_info
    import mmcv as mmengine

from prettytable import PrettyTable
import einops
import torch.nn.functional as F


def openocc_metric(pred, gt, eval_type, empty_idx, visible_mask=None):

    gt_temp = copy.deepcopy(gt)
    pred_temp = copy.deepcopy(pred)
    # ignore noise
    noise_mask = gt_temp != 255

    if eval_type == 'SC':
        gt_temp[gt_temp != empty_idx] = 1
        pred_temp[pred_temp != empty_idx] = 1
        return fast_hist(pred_temp[noise_mask], gt_temp[noise_mask], max_label=2), None

    if eval_type == 'SSC':
        hist_occ = None
        if visible_mask is not None:
            visible_mask = visible_mask[0].cpu().numpy()
            mask = noise_mask & (visible_mask != 0)
            hist_occ = fast_hist(pred_temp[mask], gt_temp[mask], max_label=17)

        hist = fast_hist(pred_temp[noise_mask], gt_temp[noise_mask], max_label=17)
        return hist, hist_occ


def fast_hist(pred, label, max_label=18):
    pred = copy.deepcopy(pred.flatten())
    label = copy.deepcopy(label.flatten())
    bin_count = np.bincount(max_label * label.astype(int) + pred, minlength=max_label ** 2)
    return bin_count[:max_label ** 2].reshape(max_label, max_label)


def collect_results_cpu(result_part, size, tmpdir=None, type='list'):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified

    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            mmengine.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmengine.mkdir_or_exist(tmpdir)

    mmengine.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()

    if rank == 0:
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmengine.load(part_file))

        # sort the results
        if type == 'list':
            ordered_results = []
            for res in part_list:
                ordered_results.extend(list(res))
            ordered_results = ordered_results[:size]

        else:
            raise NotImplementedError

        shutil.rmtree(tmpdir)

    dist.barrier()

    if rank != 0:
        return None

    return ordered_results


def cm_to_ious(cm):
    mean_ious = []
    cls_num = len(cm)
    for i in range(cls_num):
        tp = cm[i, i]
        p = cm[:, i].sum()
        g = cm[i, :].sum()
        union = p + g - tp
        if union == 0:
            mean_ious.append(0.0)
        else:
            mean_ious.append(tp / union)

    return mean_ious


def format_SC_results(mean_ious, return_dic=False):
    class_map = {
        1: 'non-empty',
    }

    x = PrettyTable()
    x.field_names = ['class', 'IoU']
    class_names = list(class_map.values())
    class_ious = mean_ious
    dic = {}

    for cls_name, cls_iou in zip(class_names, class_ious):
        dic[cls_name] = np.round(cls_iou, 3)
        x.add_row([cls_name, np.round(cls_iou, 3)])

    if return_dic:
        return x, dic
    else:
        return x


def format_SSC_results(mean_ious, return_dic=False):
    class_map = {
        0: 'free',
        1: 'barrier',
        2: 'bicycle',
        3: 'bus',
        4: 'car',
        5: 'construction_vehicle',
        6: 'motorcycle',
        7: 'pedestrian',
        8: 'traffic_cone',
        9: 'trailer',
        10: 'truck',
        11: 'driveable_surface',
        12: 'other_flat',
        13: 'sidewalk',
        14: 'terrain',
        15: 'manmade',
        16: 'vegetation',
    }

    x = PrettyTable()
    x.field_names = ['class', 'IoU']
    class_names = list(class_map.values())
    class_ious = mean_ious
    dic = {}

    for cls_name, cls_iou in zip(class_names, class_ious):
        dic[cls_name] = np.round(cls_iou, 3)
        x.add_row([cls_name, np.round(cls_iou, 3)])

    if len(mean_ious) > 1:
        valid_ious = mean_ious[1:]
        mean_iou_val = sum(valid_ious) / len(valid_ious)
    else:
        mean_iou_val = 0.0

    dic['mean'] = np.round(mean_iou_val, 3)
    x.add_row(['mean', np.round(mean_iou_val, 3)])

    if return_dic:
        return x, dic
    else:
        return x
