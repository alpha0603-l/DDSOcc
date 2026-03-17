import copy
import torch
from torch import nn
from mmcv.cnn import ConvModule, build_conv_layer
from mmengine.model import BaseModule
from mmengine.registry import MODELS, TASK_UTILS
HEADS = MODELS
def build_loss(cfg):
    return MODELS.build(cfg)
def force_fp32(apply_to=None, out_fp16=False):
    def decorator(func):
        return func
    return decorator
try:
    from mmdet3d.utils import circle_nms, draw_heatmap_gaussian, gaussian_radius
except ImportError:
    try:
        from mmdet3d.core import circle_nms, draw_heatmap_gaussian, gaussian_radius
    except ImportError:
        pass
try:
    from mmdet3d.core import xywhr2xyxyr
except ImportError:
    pass
try:
    from mmdet.models.task_modules import build_bbox_coder
except ImportError:
    def build_bbox_coder(cfg):
        return TASK_UTILS.build(cfg)
try:
    from mmdet.models.utils import multi_apply
except ImportError:
    from mmdet.core import multi_apply
try:
    from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
except ImportError:
    try:
        from mmdet3d.ops import nms_gpu
    except ImportError:
        pass

def clip_sigmoid(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)


@HEADS.register_module()
class SeparateHead(BaseModule):
    """SeparateHead for CenterHead."""
    def __init__(
            self,
            in_channels,
            heads,
            head_conv=64,
            final_kernel=1,
            init_bias=-2.19,
            conv_cfg=dict(type="Conv2d"),
            norm_cfg=dict(type="BN2d"),
            bias="auto",
            init_cfg=None,
            **kwargs,
    ):
        if init_cfg is None:
            init_cfg = dict(type="Kaiming", layer="Conv2d")

        super(SeparateHead, self).__init__(init_cfg=init_cfg)
        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.append(
                    ConvModule(
                        c_in,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=bias,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                    )
                )
                c_in = head_conv

            conv_layers.append(
                build_conv_layer(
                    conv_cfg,
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True,
                )
            )
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        for head in self.heads:
            if head == "heatmap":
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)

    def forward(self, x):
        """Forward function for SepHead."""
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


@HEADS.register_module()
class DCNSeparateHead(BaseModule):
    r"""DCNSeparateHead for CenterHead."""

    def __init__(
            self,
            in_channels,
            num_cls,
            heads,
            dcn_config,
            head_conv=64,
            final_kernel=1,
            init_bias=-2.19,
            conv_cfg=dict(type="Conv2d"),
            norm_cfg=dict(type="BN2d"),
            bias="auto",
            init_cfg=None,
            **kwargs,
    ):
        if init_cfg is None:
            init_cfg = dict(type="Kaiming", layer="Conv2d")

        super(DCNSeparateHead, self).__init__(init_cfg=init_cfg)
        if "heatmap" in heads:
            heads.pop("heatmap")
        self.feature_adapt_cls = build_conv_layer(dcn_config)
        self.feature_adapt_reg = build_conv_layer(dcn_config)

        cls_head = [
            ConvModule(
                in_channels,
                head_conv,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                bias=bias,
                norm_cfg=norm_cfg,
            ),
            build_conv_layer(
                conv_cfg,
                head_conv,
                num_cls,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
        ]
        self.cls_head = nn.Sequential(*cls_head)
        self.init_bias = init_bias
        self.task_head = SeparateHead(
            in_channels,
            heads,
            head_conv=head_conv,
            final_kernel=final_kernel,
            bias=bias,
        )

    def init_weights(self):
        """Initialize weights."""
        super().init_weights()
        self.cls_head[-1].bias.data.fill_(self.init_bias)

    def forward(self, x):
        """Forward function for DCNSepHead."""
        center_feat = self.feature_adapt_cls(x)
        reg_feat = self.feature_adapt_reg(x)

        cls_score = self.cls_head(center_feat)
        ret = self.task_head(reg_feat)
        ret["heatmap"] = cls_score

        return ret


@HEADS.register_module()
class CenterHead(BaseModule):
    """CenterHead for CenterPoint."""

    def __init__(
            self,
            in_channels=[128],
            tasks=None,
            train_cfg=None,
            test_cfg=None,
            bbox_coder=None,
            common_heads=dict(),
            loss_cls=dict(type="GaussianFocalLoss", reduction="mean"),
            loss_bbox=dict(type="L1Loss", reduction="none", loss_weight=0.25),
            separate_head=dict(type="SeparateHead", init_bias=-2.19, final_kernel=3),
            share_conv_channel=64,
            num_heatmap_convs=2,
            conv_cfg=dict(type="Conv2d"),
            norm_cfg=dict(type="BN2d"),
            bias="auto",
            norm_bbox=True,
            init_cfg=None,
            with_velocity=True,
    ):
        super(CenterHead, self).__init__(init_cfg=init_cfg)
        if separate_head is not None:
            try:
                separate_head = dict(separate_head)
            except Exception:
                pass

        if loss_cls is not None:
            try:
                loss_cls = dict(loss_cls)
            except Exception:
                pass

        if loss_bbox is not None:
            try:
                loss_bbox = dict(loss_bbox)
            except Exception:
                pass

        num_classes = [len(t) for t in tasks]
        self.class_names = [t for t in tasks]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.norm_bbox = norm_bbox

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.num_anchor_per_locs = [n for n in num_classes]
        self.with_velocity = with_velocity
        self.fp16_enabled = False

        self.shared_conv = ConvModule(
            in_channels,
            share_conv_channel,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias,
        )

        self.task_heads = nn.ModuleList()

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(num_cls, num_heatmap_convs)))
            task_head_cfg = copy.deepcopy(separate_head)
            task_head_cfg.update(
                dict(
                    in_channels=share_conv_channel,
                    heads=heads,
                    num_cls=num_cls
                )
            )
            self.task_heads.append(MODELS.build(task_head_cfg))

    def forward_single(self, x):
        """Forward function for CenterPoint."""
        ret_dicts = []
        x = self.shared_conv(x)
        for task in self.task_heads:
            ret_dicts.append(task(x))
        return ret_dicts

    def forward(self, feats, metas):
        """Forward pass."""
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        return multi_apply(self.forward_single, feats)

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map."""
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def get_targets(self, gt_bboxes_3d, gt_labels_3d):
        """Generate targets."""
        heatmaps, anno_boxes, inds, masks = multi_apply(
            self.get_targets_single, gt_bboxes_3d, gt_labels_3d
        )
        heatmaps = list(map(list, zip(*heatmaps)))
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        anno_boxes = list(map(list, zip(*anno_boxes)))
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        inds = list(map(list, zip(*inds)))
        inds = [torch.stack(inds_) for inds_ in inds]
        masks = list(map(list, zip(*masks)))
        masks = [torch.stack(masks_) for masks_ in masks]
        return heatmaps, anno_boxes, inds, masks

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample."""
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]), dim=1
        ).to(device)
        max_objs = self.train_cfg["max_objs"] * self.train_cfg["dense_reg"]
        grid_size = torch.tensor(self.train_cfg["grid_size"])
        pc_range = torch.tensor(self.train_cfg["point_cloud_range"])
        voxel_size = torch.tensor(self.train_cfg["voxel_size"])

        feature_map_size = torch.div(
            grid_size[:2],
            self.train_cfg["out_size_factor"],
            rounding_mode="trunc",
        )

        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append(
                [
                    torch.where(gt_labels_3d == class_name.index(i) + flag)
                    for i in class_name
                ]
            )
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1], feature_map_size[0])
            )
            if self.with_velocity:
                anno_box = gt_bboxes_3d.new_zeros((max_objs, 10), dtype=torch.float32)
            else:
                anno_box = gt_bboxes_3d.new_zeros((max_objs, 8), dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.train_cfg["out_size_factor"]
                length = length / voxel_size[1] / self.train_cfg["out_size_factor"]

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width), min_overlap=self.train_cfg["gaussian_overlap"]
                    )
                    radius = max(self.train_cfg["min_radius"], int(radius))

                    x, y, z = (
                        task_boxes[idx][k][0],
                        task_boxes[idx][k][1],
                        task_boxes[idx][k][2],
                    )

                    coor_x = (
                            (x - pc_range[0])
                            / voxel_size[0]
                            / self.train_cfg["out_size_factor"]
                    )
                    coor_y = (
                            (y - pc_range[1])
                            / voxel_size[1]
                            / self.train_cfg["out_size_factor"]
                    )

                    center = torch.tensor(
                        [coor_x, coor_y], dtype=torch.float32, device=device
                    )
                    center_int = center.to(torch.int32)

                    if not (
                            0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]
                    ):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int[[1, 0]], radius)
                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert (
                            x * feature_map_size[1] + y
                            < feature_map_size[0] * feature_map_size[1]
                    )

                    ind[new_idx] = x * feature_map_size[1] + y

                    mask[new_idx] = 1
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    if self.norm_bbox:
                        box_dim = box_dim.log()
                    if self.with_velocity:
                        vx, vy = task_boxes[idx][k][7:]
                        anno_box[new_idx] = torch.cat(
                            [
                                center - torch.tensor([x, y], device=device),
                                z.unsqueeze(0),
                                box_dim,
                                torch.sin(rot).unsqueeze(0),
                                torch.cos(rot).unsqueeze(0),
                                vx.unsqueeze(0),
                                vy.unsqueeze(0),
                            ]
                        )
                    else:
                        anno_box[new_idx] = torch.cat(
                            [
                                center - torch.tensor([x, y], device=device),
                                z.unsqueeze(0),
                                box_dim,
                                torch.sin(rot).unsqueeze(0),
                                torch.cos(rot).unsqueeze(0),
                            ]
                        )

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks

    @force_fp32(apply_to=("preds_dicts"))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead."""
        heatmaps, anno_boxes, inds, masks = self.get_targets(gt_bboxes_3d, gt_labels_3d)
        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            preds_dict[0]["heatmap"] = clip_sigmoid(preds_dict[0]["heatmap"])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_heatmap = self.loss_cls(
                preds_dict[0]["heatmap"], heatmaps[task_id], avg_factor=max(num_pos, 1)
            )
            target_box = anno_boxes[task_id]
            if self.with_velocity:
                preds_dict[0]["anno_box"] = torch.cat(
                    (
                        preds_dict[0]["reg"],
                        preds_dict[0]["height"],
                        preds_dict[0]["dim"],
                        preds_dict[0]["rot"],
                        preds_dict[0]["vel"],
                    ),
                    dim=1,
                )
            else:
                preds_dict[0]["anno_box"] = torch.cat(
                    (
                        preds_dict[0]["reg"],
                        preds_dict[0]["height"],
                        preds_dict[0]["dim"],
                        preds_dict[0]["rot"],
                    ),
                    dim=1,
                )
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict[0]["anno_box"].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            code_weights = self.train_cfg.get("code_weights", None)
            bbox_weights = mask * mask.new_tensor(code_weights)
            loss_bbox = self.loss_bbox(
                pred, target_box, bbox_weights, avg_factor=(num + 1e-4)
            )
            loss_dict[f"heatmap/task{task_id}"] = loss_heatmap
            loss_dict[f"bbox/task{task_id}"] = loss_bbox
        return loss_dict

    @force_fp32(apply_to=("preds_dicts"))
    def get_bboxes(self, preds_dicts, metas, img=None, rescale=False):
        """Generate bboxes from bbox head predictions."""

        if not isinstance(self.test_cfg["nms_type"], list):
            nms_types = [self.test_cfg["nms_type"] for _ in range(len(preds_dicts))]
        else:
            nms_types = self.test_cfg["nms_type"]

        if "nms_scale" in self.test_cfg:
            if not isinstance(self.test_cfg["nms_scale"], list):
                nms_scales = [
                    [
                        self.test_cfg["nms_scale"]
                        for _ in range(self.num_classes[task_id])
                    ]
                    for task_id in range(len(preds_dicts))
                ]
            else:
                nms_scales = self.test_cfg["nms_scale"]
        else:
            nms_scales = [
                [1.0 for _ in range(self.num_classes[task_id])]
                for task_id in range(len(preds_dicts))
            ]

        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict[0]["heatmap"].shape[0]
            batch_heatmap = preds_dict[0]["heatmap"].sigmoid()

            batch_reg = preds_dict[0]["reg"]
            batch_hei = preds_dict[0]["height"]

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]["dim"])
            else:
                batch_dim = preds_dict[0]["dim"]

            batch_rots = preds_dict[0]["rot"][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]["rot"][:, 1].unsqueeze(1)

            if "vel" in preds_dict[0]:
                batch_vel = preds_dict[0]["vel"]
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id,
            )
            batch_reg_preds = [box["bboxes"] for box in temp]
            batch_cls_preds = [box["scores"] for box in temp]
            batch_cls_labels = [box["labels"] for box in temp]
            if nms_types[task_id] == "circle":
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]["bboxes"]
                    scores = temp[i]["scores"]
                    labels = temp[i]["labels"]
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg["min_radius"][task_id],
                            post_max_size=self.test_cfg["post_max_size"],
                        ),
                        dtype=torch.long,
                        device=boxes.device,
                    )

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(
                        num_class_with_bg,
                        batch_cls_preds,
                        batch_reg_preds,
                        batch_cls_labels,
                        metas,
                        nms_scales[task_id],
                    )
                )

        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == "bboxes":
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = metas[i]["box_type_3d"](bboxes, self.bbox_coder.code_size)
                elif k == "scores":
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == "labels":
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])
        return ret_list

    def get_task_detections(
            self,
            num_class_with_bg,
            batch_cls_preds,
            batch_reg_preds,
            batch_cls_labels,
            metas,
            nms_scale=1.0,
    ):
        """Rotate nms for each task."""
        predictions_dicts = []
        post_center_range = self.test_cfg["post_center_limit_range"]
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds[0].dtype,
                device=batch_reg_preds[0].device,
            )

        for i, (box_preds, cls_preds, cls_labels) in enumerate(
                zip(batch_reg_preds, batch_cls_preds, batch_cls_labels)
        ):
            if num_class_with_bg == 1:
                top_scores = cls_preds.squeeze(-1)
                top_labels = torch.zeros(
                    cls_preds.shape[0], device=cls_preds.device, dtype=torch.long
                )

            else:
                top_labels = cls_labels.long()
                top_scores = cls_preds.squeeze(-1)

            if self.test_cfg["score_threshold"] > 0.0:
                thresh = torch.tensor(
                    [self.test_cfg["score_threshold"]], device=cls_preds.device
                ).type_as(cls_preds)
                top_scores_keep = top_scores >= thresh
                top_scores = top_scores.masked_select(top_scores_keep)

            if top_scores.shape[0] != 0:
                if self.test_cfg["score_threshold"] > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]

                bev_box = metas[i]["box_type_3d"](
                    box_preds[:, :], self.bbox_coder.code_size
                ).bev
                for cls, scale in enumerate(nms_scale):
                    cur_bev_box = bev_box[top_labels == cls]
                    cur_bev_box[:, [2, 3]] *= scale
                    bev_box[top_labels == cls] = cur_bev_box
                boxes_for_nms = xywhr2xyxyr(bev_box)

                selected = nms_gpu(
                    boxes_for_nms,
                    top_scores,
                    thresh=self.test_cfg["nms_thr"],
                    pre_maxsize=self.test_cfg["pre_max_size"],
                    post_max_size=self.test_cfg["post_max_size"],
                )
            else:
                selected = []

            selected_boxes = box_preds[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]

            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >= post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <= post_center_range[3:]).all(1)
                    predictions_dict = dict(
                        bboxes=final_box_preds[mask],
                        scores=final_scores[mask],
                        labels=final_labels[mask],
                    )
                else:
                    predictions_dict = dict(
                        bboxes=final_box_preds, scores=final_scores, labels=final_labels
                    )
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = dict(
                    bboxes=torch.zeros(
                        [0, self.bbox_coder.code_size], dtype=dtype, device=device
                    ),
                    scores=torch.zeros([0], dtype=dtype, device=device),
                    labels=torch.zeros([0], dtype=top_labels.dtype, device=device),
                )

            predictions_dicts.append(predictions_dict)
        return predictions_dicts
