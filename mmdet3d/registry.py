from mmengine.registry import Registry
MODELS = Registry('model', scope='mmdet3d')
DATASETS = Registry('dataset', scope='mmdet3d')
TRANSFORMS = Registry('transform', scope='mmdet3d')
PIPELINES = TRANSFORMS
