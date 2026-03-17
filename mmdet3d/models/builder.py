from mmdet.registry import MODELS
import warnings
BACKBONES = MODELS
HEADS = MODELS
LOSSES = MODELS
NECKS = MODELS
FUSIONMODELS = MODELS
VTRANSFORMS = MODELS
FUSERS = MODELS
CUSTOMS = MODELS

def build_backbone(cfg):
    return MODELS.build(cfg)

def build_neck(cfg):
    return MODELS.build(cfg)

def build_vtransform(cfg):
    return MODELS.build(cfg)

def build_fuser(cfg):
    return MODELS.build(cfg)

def build_custom(cfg):
    return MODELS.build(cfg)

def build_head(cfg):
    return MODELS.build(cfg)

def build_loss(cfg):
    return MODELS.build(cfg)

def build_fusion_model(cfg, train_cfg=None, test_cfg=None):

    return MODELS.build(cfg)

def build_model(cfg, train_cfg=None, test_cfg=None):
    return build_fusion_model(cfg, train_cfg=train_cfg, test_cfg=test_cfg)

__all__ = [
    'BACKBONES', 'HEADS', 'LOSSES', 'NECKS',
    'FUSIONMODELS', 'VTRANSFORMS', 'FUSERS', 'CUSTOMS',
    'build_backbone', 'build_neck', 'build_vtransform', 'build_fuser',
    'build_custom', 'build_head', 'build_loss',
    'build_fusion_model', 'build_model'
]
