import copy
import os
import mmcv
import numpy as np
try:
    from mmdet3d.structures.ops import box_np_ops
except ImportError:
    try:
        from mmdet3d.core.bbox import box_np_ops
    except ImportError:
        pass
from mmdet3d.registry import TRANSFORMS
PIPELINES = TRANSFORMS

try:
    from ..builder import OBJECTSAMPLERS
except ImportError:
    from mmengine.registry import Registry

    OBJECTSAMPLERS = Registry('object_samplers')

import mmengine
from .utils import box_collision_test


class BatchSampler:
    """Class for sampling specific category of ground truths."""

    def __init__(
            self, sampled_list, name=None, epoch=None, shuffle=True, drop_reminder=False
    ):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num):
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx:].copy()
            self._reset()
        else:
            ret = self._indices[self._idx: self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        assert self._name is not None
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]


@OBJECTSAMPLERS.register_module()
class DataBaseSampler:
    """Class for sampling data from the ground truth database."""

    def __init__(
            self,
            info_path,
            dataset_root,
            rate,
            prepare,
            sample_groups,
            classes=None,
            points_loader=dict(
                type="LoadPointsFromFile",
                coord_type="LIDAR",
                load_dim=4,
                use_dim=[0, 1, 2, 3],
            ),
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.info_path = info_path
        self.rate = rate
        self.prepare = prepare
        self.classes = classes
        if classes:
            self.cat2label = {name: i for i, name in enumerate(classes)}
            self.label2cat = {i: name for i, name in enumerate(classes)}
        else:
            self.cat2label = {}
            self.label2cat = {}
        if isinstance(points_loader, dict):
            self.points_loader = TRANSFORMS.build(points_loader)
        else:
            self.points_loader = points_loader
        if hasattr(mmengine, 'load'):
            load_func = mmengine.load
        else:
            load_func = mmcv.load

        db_infos = load_func(info_path)
        try:
            from mmengine.logging import MMEngineLogger
            logger = MMEngineLogger.get_current_instance()
        except ImportError:
            logger = None

        if logger:
            for k, v in db_infos.items():
                logger.info(f"load {len(v)} {k} database infos")

        for prep_func, val in prepare.items():
            db_infos = getattr(self, prep_func)(db_infos, val)

        if logger:
            logger.info("After filter database:")
            for k, v in db_infos.items():
                logger.info(f"load {len(v)} {k} database infos")

        self.db_infos = db_infos
        self.sample_groups = []
        for name, num in sample_groups.items():
            self.sample_groups.append({name: int(num)})

        self.group_db_infos = self.db_infos
        self.sample_classes = []
        self.sample_max_nums = []
        for group_info in self.sample_groups:
            self.sample_classes += list(group_info.keys())
            self.sample_max_nums += list(group_info.values())

        self.sampler_dict = {}
        for k, v in self.group_db_infos.items():
            self.sampler_dict[k] = BatchSampler(v, k, shuffle=True)

    @staticmethod
    def filter_by_difficulty(db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info for info in dinfos if info["difficulty"] not in removed_difficulty
            ]
        return new_db_infos

    @staticmethod
    def filter_by_min_points(db_infos, min_gt_points_dict):
        for name, min_num in min_gt_points_dict.items():
            min_num = int(min_num)
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info["num_points_in_gt"] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos

    def sample_all(self, gt_bboxes, gt_labels, img=None, gt_bboxes_2d=None):
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(
                self.sample_classes, self.sample_max_nums
        ):
            class_label = self.cat2label[class_name]
            sampled_num = int(
                max_sample_num - np.sum([n == class_label for n in gt_labels])
            )
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes = []
        avoid_coll_boxes = gt_bboxes

        for class_name, sampled_num in zip(self.sample_classes, sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class_v2(
                    class_name, sampled_num, avoid_coll_boxes
                )

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]["box3d_lidar"][np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s["box3d_lidar"] for s in sampled_cls], axis=0
                        )

                    sampled_gt_bboxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0
                    )

        ret = None
        if len(sampled) > 0:
            sampled_gt_bboxes = np.concatenate(sampled_gt_bboxes, axis=0)

            s_points_list = []
            count = 0
            for info in sampled:
                file_path = (
                    os.path.join(self.dataset_root, info["path"])
                    if self.dataset_root
                    else info["path"]
                )
                results = dict(lidar_path=file_path)
                s_points = self.points_loader(results)["points"]
                s_points.translate(info["box3d_lidar"][:3])

                count += 1
                s_points_list.append(s_points)

            gt_labels = np.array(
                [self.cat2label[s["name"]] for s in sampled], dtype=np.int64
            )
            ret = {
                "gt_labels_3d": gt_labels,
                "gt_bboxes_3d": sampled_gt_bboxes,
                "points": s_points_list[0].cat(s_points_list),
                "group_ids": np.arange(
                    gt_bboxes.shape[0], gt_bboxes.shape[0] + len(sampled)
                ),
            }
            if gt_bboxes_2d is not None and img is not None:
                pass

        return ret

    def sample_class_v2(self, name, num, gt_bboxes):
        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_bboxes.shape[0]
        num_sampled = len(sampled)
        try:
            if 'box_np_ops' not in globals():
                from mmdet3d.structures.ops import box_np_ops

            gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
                gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6]
            )
        except (ImportError, NameError):
            from mmdet3d.core.bbox import box_np_ops
            gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
                gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6]
            )

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)
        boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_bboxes.shape[0]:]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6]
        )

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)

        coll_mat = box_collision_test(total_bv, total_bv)

        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
        return valid_samples
