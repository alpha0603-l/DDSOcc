import platform
import copy
import torch
from functools import partial
from torch.utils.data import DataLoader
from mmengine.registry import Registry
from mmengine.dataset import ClassBalancedDataset, ConcatDataset, RepeatDataset

try:
    from mmdet.registry import DATASETS
except ImportError:
    from mmdet.datasets import DATASETS
OBJECTSAMPLERS = Registry("object_sampler")
PIPELINES = Registry("pipeline")

if platform.system() != "Windows":
    import resource
    try:
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        base_soft_limit = rlimit[0]
        hard_limit = rlimit[1]
        soft_limit = min(max(4096, base_soft_limit), hard_limit)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))
    except Exception:
        pass


def build_dataset(cfg, default_args=None):
    try:
        from mmdet3d.datasets.dataset_wrappers import CBGSDataset
    except ImportError:
        pass

    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg["type"] == "ConcatDataset":
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg["datasets"]],
            cfg.get("separate_eval", True),
        )
    elif cfg["type"] == "RepeatDataset":
        dataset = RepeatDataset(build_dataset(cfg["dataset"], default_args), cfg["times"])
    elif cfg["type"] == "ClassBalancedDataset":
        dataset = ClassBalancedDataset(
            build_dataset(cfg["dataset"], default_args), cfg["oversample_thr"]
        )
    elif cfg["type"] == "CBGSDataset":
        dataset = CBGSDataset(build_dataset(cfg["dataset"], default_args))
    elif isinstance(cfg.get("ann_file"), (list, tuple)):
        cfg_ = copy.deepcopy(cfg)
        return DATASETS.build(cfg_, default_args=default_args)
    else:
        dataset = DATASETS.build(cfg, default_args=default_args)

    return dataset

def fast_collate_fn(batch):
    if not isinstance(batch, list):
        return batch

    if len(batch) == 0:
        return {}

    elem = batch[0]
    if isinstance(elem, dict):
        return {key: [d[key] for d in batch] for key in elem}
    else:
        return batch


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     **kwargs):

    sampler = None
    if dist:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False

    init_fn = partial(worker_init_fn, num_workers=workers_per_gpu, rank=0, seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=samples_per_gpu,
        sampler=sampler,
        num_workers=workers_per_gpu,
        collate_fn=fast_collate_fn,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        **kwargs
    )

    return data_loader

def worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    import numpy as np
    import random
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
