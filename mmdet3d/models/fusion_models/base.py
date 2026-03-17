from mmengine.model import BaseModel
from abc import ABCMeta
__all__ = ["Base3DFusionModel"]
class Base3DFusionModel(BaseModel, metaclass=ABCMeta):

    def __init__(self, init_cfg=None, data_preprocessor=None):
        super().__init__(init_cfg=init_cfg, data_preprocessor=data_preprocessor)
        self.fp16_enabled = False

    def forward(self, inputs, data_samples, mode='tensor'):
        pass
