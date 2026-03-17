import numpy as np
from mmengine.registry import DATASETS
from mmengine.dataset import BaseDataset


@DATASETS.register_module(name='CBGSDataset', force=True)
class CBGSDataset(BaseDataset):

    def __init__(self, dataset, **kwargs):

        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        else:
            self.dataset = dataset
        self.resample = getattr(self.dataset, "resample", True)
        if hasattr(self.dataset, 'metainfo'):
            self.CLASSES = self.dataset.metainfo.get('classes', None)
        else:
            self.CLASSES = getattr(self.dataset, "CLASSES", None)
        if self.CLASSES:
            self._metainfo = dict(classes=self.CLASSES)
        else:
            self._metainfo = dict(classes=[])

        if self.CLASSES is None:
            if hasattr(self.dataset, 'CLASSES'):
                self.CLASSES = self.dataset.CLASSES
                self._metainfo = dict(classes=self.CLASSES)
            else:
                self.CLASSES = []

        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        if self.resample:
            self.sample_indices = self._get_sample_indices()
        else:
            self.sample_indices = []
        if hasattr(self.dataset, "flag"):
            if self.resample:
                if len(self.dataset.flag) > 0:
                    self.flag = np.array(
                        [self.dataset.flag[ind] for ind in self.sample_indices], dtype=np.uint8
                    )
                else:
                    self.flag = np.zeros(len(self.sample_indices), dtype=np.uint8)
            else:
                self.flag = self.dataset.flag
        else:
            length = len(self.sample_indices) if self.resample else len(self.dataset)
            self.flag = np.zeros(length, dtype=np.uint8)
        super().__init__(lazy_init=False, serialize_data=False)
    def get_data_info(self, idx):
        if self.resample:
            idx = self.sample_indices[idx]
        return self.dataset.get_data_info(idx)
    def full_init(self):
        if hasattr(self.dataset, 'full_init'):
            self.dataset.full_init()
        if hasattr(self.dataset, "flag") and self.resample:
            if len(self.dataset.flag) > 0:
                self.flag = np.array(
                    [self.dataset.flag[ind] for ind in self.sample_indices], dtype=np.uint8
                )
    def set_epoch(self, epoch):
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)

    def _get_sample_indices(self):
        if hasattr(self.dataset, 'full_init'):
            self.dataset.full_init()
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        if not hasattr(self.dataset, 'get_cat_ids'):
            print(f"[CBGS Warning] Inner dataset {type(self.dataset)} no 'get_cat_ids'. Disable CBGS.")
            return list(range(len(self.dataset)))

        for idx in range(len(self.dataset)):
            sample_cat_ids = self.dataset.get_cat_ids(idx)
            valid_ids = [c for c in sample_cat_ids if c in class_sample_idxs]
            for cat_id in valid_ids:
                class_sample_idxs[cat_id].append(idx)

        duplicated_samples = sum([len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / (duplicated_samples + 1e-6) for k, v in class_sample_idxs.items()
        }

        sample_indices = []
        frac = 1.0 / (len(self.CLASSES) + 1e-6)
        ratios = [frac / (v + 1e-6) for v in class_distribution.values()]

        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            if len(cls_inds) > 0:
                sample_indices += np.random.choice(
                    cls_inds, int(len(cls_inds) * ratio)
                ).tolist()

        if not sample_indices:
            sample_indices = list(range(len(self.dataset)))

        return sample_indices

    def __getitem__(self, idx):
        if self.resample:
            ori_idx = self.sample_indices[idx]
            return self.dataset[ori_idx]
        else:
            return self.dataset[idx]

    def __len__(self):
        if self.resample:
            return len(self.sample_indices)
        else:
            return len(self.dataset)

    @property
    def metainfo(self):
        return self._metainfo
    def __getattr__(self, name):
        if name in ('_metainfo', 'dataset', 'sample_indices', 'resample', 'CLASSES', 'cat2id', 'flag'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.dataset, name)
