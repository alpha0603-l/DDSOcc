import numpy as np
import torch
from torch import nn as nn
from torch.utils.checkpoint import checkpoint
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.cnn.bricks import ConvModule
from mmengine.model import BaseModule
from mmdet.registry import MODELS

__all__ = ["SECONDFPN", "FPN_LSS", "LSSFPN3D"]

@MODELS.register_module()
class SECONDFPN(BaseModule):

    def __init__(
        self,
        in_channels=[128, 128, 256],
        out_channels=[256, 256, 256],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type="deconv", bias=False),
        conv_cfg=dict(type="Conv2d", bias=False),
        use_conv_for_no_stride=False,
        init_cfg=None,
    ):
        super(SECONDFPN, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i],
                )
            else:
                stride = int(np.round(1 / stride))
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride,
                )

            deblock = nn.Sequential(
                upsample_layer,
                build_norm_layer(norm_cfg, out_channel)[1],
                nn.ReLU(inplace=True),
            )
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)

        if init_cfg is None:
            self.init_cfg = [
                dict(type="Kaiming", layer="ConvTranspose2d"),
                dict(type="Constant", layer="NaiveSyncBatchNorm2d", val=1.0),
            ]
    def forward(self, x):
        """Forward function."""
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return [out]

@MODELS.register_module()
class FPN_LSS(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 use_input_conv=False,
                 init_cfg=None):
        super(FPN_LSS, self).__init__(init_cfg=init_cfg)
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.out_channels = out_channels
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)

        channels_factor = 2 if self.extra_upsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * channels_factor, kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * channels_factor, out_channels * channels_factor, kernel_size=3,
                      padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
        )

        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=extra_upsample, mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels * channels_factor, out_channels, kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
            )

        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(lateral, lateral, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral)[1],
                nn.ReLU(inplace=True)
            )

    def forward(self, feats):

        x2, x1 = feats[self.input_feature_index[0]], feats[self.input_feature_index[1]]
        if self.lateral:
            x2 = self.lateral_conv(x2)
        x1 = self.up(x1)    # (B, C3, H, W)
        x1 = torch.cat([x2, x1], dim=1)     # (B, C1+C3, H, W)
        x = self.conv(x1)   # (B, C', H, W)
        if self.extra_upsample:
            x = self.up2(x)     # (B, C_out, 2*H, 2*W)
        return x


@MODELS.register_module()  # [修复] NECKS -> MODELS
class LSSFPN3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 with_cp=False):
        super().__init__()
        self.up1 = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)
        self.up2 = nn.Upsample(
            scale_factor=4, mode='trilinear', align_corners=True)

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU', inplace=True))
        self.with_cp = with_cp

    def forward(self, feats):
        x_8, x_16, x_32 = feats
        x_16 = self.up1(x_16)       # (B, 2C, Dz, Dy, Dx)
        x_32 = self.up2(x_32)       # (B, 4C, Dz, Dy, Dx)
        x = torch.cat([x_8, x_16, x_32], dim=1)     # (B, 7C, Dz, Dy, Dx)
        if self.with_cp:
            x = checkpoint(self.conv, x)
        else:
            x = self.conv(x)    # (B, C, Dz, Dy, Dx)
        return x
