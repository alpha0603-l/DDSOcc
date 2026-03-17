from typing import List, Tuple
import torch
from torch import nn
try:
    from mmcv.cnn.resnet import BasicBlock, make_res_layer
except ImportError:
    from mmdet.models.backbones.resnet import BasicBlock
    from mmcv.cnn import make_res_layer
from mmengine.model import BaseModule
from mmdet.registry import MODELS

__all__ = ["GeneralizedResNet"]


@MODELS.register_module()
class GeneralizedResNet(BaseModule):
    def __init__(
            self,
            in_channels: int,
            blocks: List[Tuple[int, int, int]],
            init_cfg=None
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.blocks_config = blocks
        _layers = []
        for num_blocks, out_channels, stride in self.blocks_config:
            res_layer = make_res_layer(
                BasicBlock,
                in_channels,
                out_channels,
                num_blocks,
                stride=stride,
                dilation=1,
            )
            in_channels = out_channels
            _layers.append(res_layer)
        self.layers = nn.ModuleList(_layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for module in self.layers:
            x = module(x)
            outputs.append(x)
        return outputs
