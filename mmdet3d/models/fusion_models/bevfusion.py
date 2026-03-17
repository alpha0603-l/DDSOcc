from typing import Any, Dict, List, Union
import torch
from torch import nn
from torch.nn import functional as F
from mmengine.registry import MODELS
def build_backbone(cfg): return MODELS.build(cfg)
def build_neck(cfg): return MODELS.build(cfg)
def build_vtransform(cfg): return MODELS.build(cfg)
def build_fuser(cfg): return MODELS.build(cfg)
def build_head(cfg): return MODELS.build(cfg)
try:
    from mmdet3d.models.layers import Voxelization, DynamicScatter
except ImportError:
    try:
        from mmdet3d.ops import Voxelization, DynamicScatter
    except ImportError:
        pass
def auto_fp16(apply_to=None, out_fp32=False):
    def decorator(func):
        return func
    return decorator
def force_fp32(apply_to=None, out_fp16=False):
    def decorator(func):
        return func
    return decorator
from .base import Base3DFusionModel
__all__ = ["BEVFusion"]


@MODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
            self,
            encoders: Dict[str, Any],
            fuser: Dict[str, Any],
            decoder: Dict[str, Any],
            heads: Dict[str, Any],
            **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
            if encoders["camera"].get("bev_backbone") is not None:
                self.encoders["camera"]["bev_backbone"] = build_backbone(encoders["camera"]["bev_backbone"])
            if encoders["camera"].get("bev_neck") is not None:
                self.encoders["camera"]["bev_neck"] = build_neck(encoders["camera"]["bev_neck"])

        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)
            if encoders["lidar"].get("bev_backbone") is not None:
                self.encoders["lidar"]["bev_backbone"] = build_backbone(encoders["lidar"]["bev_backbone"])
            if encoders["lidar"].get("bev_neck") is not None:
                self.encoders["lidar"]["bev_neck"] = build_neck(encoders["lidar"]["bev_neck"])

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        if decoder is not None:
            self.decoder = nn.ModuleDict(
                {
                    "backbone": build_backbone(decoder["backbone"]),
                    "neck": build_neck(decoder["neck"]),
                }
            )
        else:
            self.decoder = None

        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            try:
                self.encoders["camera"]["backbone"].init_weights()
            except AttributeError:
                pass

    def extract_camera_features(
            self,
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            **kwargs,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            **kwargs,
        )

        if "bev_backbone" in self.encoders["camera"]:
            x = self.encoders["camera"]["bev_backbone"](x)
        if "bev_neck" in self.encoders["camera"]:
            x = self.encoders["camera"]["bev_neck"](x)

        return x

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)

        if "bev_backbone" in self.encoders["lidar"]:
            x = self.encoders["lidar"]["bev_backbone"](x)
        if "bev_neck" in self.encoders["lidar"]:
            x = self.encoders["lidar"]["bev_neck"](x)

        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes
    @auto_fp16(apply_to=("img", "points"))
    def forward(self, *args, mode='tensor', **kwargs):
        def unpack_dc(data):
            if isinstance(data, list):
                return [unpack_dc(item) for item in data]
            if hasattr(data, 'data'):
                return data.data
            return data

        for k, v in kwargs.items():
            kwargs[k] = unpack_dc(v)
        KEYS_TO_STACK = [
            'img',
            'camera2ego', 'lidar2ego', 'lidar2camera', 'lidar2image',
            'camera_intrinsics', 'camera2lidar',
            'img_aug_matrix', 'lidar_aug_matrix', 'occ_aug_matrix',
            'voxel_semantics', 'mask_camera', 'gt_masks_bev'
        ]

        for key in KEYS_TO_STACK:
            if key in kwargs and isinstance(kwargs[key], list):
                if len(kwargs[key]) > 0 and isinstance(kwargs[key][0], torch.Tensor):
                    kwargs[key] = torch.stack(kwargs[key])
        device = None
        if hasattr(self, 'encoders') and 'camera' in self.encoders:
            try:
                device = next(self.encoders['camera']['backbone'].parameters()).device
            except StopIteration:
                pass

        if device is not None:
            for key in KEYS_TO_STACK:
                if key in kwargs and isinstance(kwargs[key], torch.Tensor):
                    kwargs[key] = kwargs[key].to(device)

            if 'points' in kwargs and isinstance(kwargs['points'], list):
                kwargs['points'] = [p.to(device) if isinstance(p, torch.Tensor) else p for p in kwargs['points']]

            gt_keys = ['gt_labels_3d', 'gt_bboxes_3d']
            for k in gt_keys:
                if k in kwargs:
                    val = kwargs[k]
                    if isinstance(val, list):
                        kwargs[k] = [v.to(device) if hasattr(v, 'to') else v for v in val]
        if 'img_metas' in kwargs and 'metas' not in kwargs:
            kwargs['metas'] = kwargs['img_metas']

        return self.forward_single(*args, **kwargs)

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
            self,
            img,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            metas,
            gt_masks_bev=None,
            gt_bboxes_3d=None,
            gt_labels_3d=None,
            **kwargs,
    ):
        features = []
        for sensor in (self.encoders if self.training else list(self.encoders.keys())[::-1]):
            if sensor == "camera":
                kwargs.pop('img_metas', None)

                feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                    **kwargs,
                )
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        if not self.training:
            features = features[::-1]

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        if self.decoder is not None:
            x = self.decoder["backbone"](x)
            x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for head_type, head in self.heads.items():
                if head_type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif head_type == "map":
                    losses = head(x, gt_masks_bev)
                elif head_type == "occ":
                    occ_pred = head(x, lidar_aug_matrix, lidar2ego, kwargs.get('occ_aug_matrix'))
                    losses = head.loss(occ_pred, kwargs.get('voxel_semantics'), kwargs.get('mask_camera'))
                else:
                    raise ValueError(f"unsupported head: {head_type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{head_type}/{name}"] = val * self.loss_scale[head_type]
                    else:
                        outputs[f"stats/{head_type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for head_type, head in self.heads.items():
                if head_type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif head_type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                elif head_type == "occ":
                    occ_pred = head(x, lidar_aug_matrix, lidar2ego, kwargs.get('occ_aug_matrix'))
                    occ_pred = head.get_occ(occ_pred)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "occ_pred": occ_pred[k],
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {head_type}")
            return outputs
