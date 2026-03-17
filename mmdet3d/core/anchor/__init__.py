from mmdet.registry import TASK_UTILS

def build_prior_generator(cfg, default_args=None):
    return TASK_UTILS.build(cfg, default_args=default_args)

try:
    from .anchor_3d_generator import AlignedAnchor3DRangeGenerator, Anchor3DRangeGenerator
    __all__ = [
        'AlignedAnchor3DRangeGenerator', 'Anchor3DRangeGenerator',
        'build_prior_generator'
    ]
except ImportError:
    __all__ = ['build_prior_generator']
