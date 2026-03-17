import tempfile
from os import path as osp
import warnings
import mmcv
import numpy as np
from torch.utils.data import Dataset
from mmdet.registry import DATASETS
from mmcv.transforms import Compose
try:
    from mmdet3d.structures import get_box_type
except ImportError:
    from ..core.bbox import get_box_type

from .utils import extract_result_dict

@DATASETS.register_module()
class Custom3DDataset(Dataset):

    def __init__(
            self,
            dataset_root,
            ann_file,
            pipeline=None,
            classes=None,
            modality=None,
            box_type_3d="LiDAR",
            filter_empty_gt=True,
            test_mode=False,
            data_prefix=dict(),
            metainfo=None,
            lazy_init=False,
            **kwargs,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.metainfo = metainfo if metainfo is not None else dict()
        self.CLASSES = self.get_classes(classes)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.data_infos = self.load_annotations(self.ann_file)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        if not self.test_mode:
            self._set_group_flag()

        self.epoch = -1

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self, "pipeline"):
            transforms = getattr(self.pipeline, "transforms", self.pipeline)
            if isinstance(transforms, list):
                for transform in transforms:
                    if hasattr(transform, "set_epoch"):
                        transform.set_epoch(epoch)

    def load_annotations(self, ann_file):
        try:
            return mmcv.load(ann_file)
        except AttributeError:
            import mmengine
            return mmengine.load(ann_file)

    def get_data_info(self, index):
        """Get data info according to the given index."""
        info = self.data_infos[index]
        if isinstance(info, dict) and 'point_cloud' in info:
            sample_idx = info["point_cloud"]["lidar_idx"]
            pts_path = info["pts_path"]
        else:
            sample_idx = info.get('sample_idx', index)
            pts_path = info.get('lidar_path', info.get('pts_path'))

        lidar_path = osp.join(self.dataset_root, pts_path)

        input_dict = dict(
            lidar_path=lidar_path, sample_idx=sample_idx, file_name=lidar_path
        )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict["ann_info"] = annos
            if self.filter_empty_gt and ~(annos["gt_labels_3d"] != -1).any():
                return None
        return input_dict

    def get_ann_info(self, index):
        return self.data_infos[index]['ann_info']

    def pre_pipeline(self, results):
        """Initialization before data preparation."""
        results["img_fields"] = []
        results["bbox3d_fields"] = []
        results["pts_mask_fields"] = []
        results["pts_seg_fields"] = []
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []
        results["box_type_3d"] = self.box_type_3d
        results["box_mode_3d"] = self.box_mode_3d

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and (
                example is None or ~(example["gt_labels_3d"].data != -1).any()
        ):
            return None
        return example

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset."""
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            try:
                class_names = mmcv.list_from_file(classes)
            except AttributeError:
                import mmengine
                class_names = mmengine.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f"Unsupported type {type(classes)} of classes.")

        return class_names

    def format_results(self, outputs, pklfile_prefix=None, submission_prefix=None):
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, "results")
            out = f"{pklfile_prefix}.pkl"

        try:
            mmcv.dump(outputs, out)
        except AttributeError:
            import mmengine
            mmengine.dump(outputs, out)

        return outputs, tmp_dir

    def _extract_data(self, index, pipeline, key, load_annos=False):
        assert pipeline is not None, "data loading pipeline is not provided"
        if load_annos:
            original_test_mode = self.test_mode
            self.test_mode = False
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = pipeline(input_dict)
        if isinstance(key, str):
            data = extract_result_dict(example, key)
        else:
            data = [extract_result_dict(example, k) for k in key]
        if load_annos:
            self.test_mode = original_test_mode

        return data

    def __len__(self):
        return len(self.data_infos)

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
