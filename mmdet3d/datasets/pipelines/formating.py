import numpy as np
import torch
from mmengine.registry import TRANSFORMS
try:
    from mmcv.parallel import DataContainer as DC
except ImportError:
    class DC:
        def __init__(self, data, stack=False, padding_value=0, cpu_only=False, pad_dims=2):
            self.data = data
            self.stack = stack
            self.padding_value = padding_value
            self.cpu_only = cpu_only
            self.pad_dims = pad_dims

        def __repr__(self):
            return f"{self.__class__.__name__}({self.data})"
try:
    from mmcv.transforms import to_tensor
except ImportError:
    try:
        from mmdet.datasets.pipelines import to_tensor
    except ImportError:
        def to_tensor(data):
            return torch.tensor(data)
try:
    from mmdet3d.structures import BaseInstance3DBoxes
    from mmdet3d.structures.points import BasePoints
except ImportError:
    from mmdet3d.core.bbox import BaseInstance3DBoxes
    from mmdet3d.core.points import BasePoints

PIPELINES = TRANSFORMS

@PIPELINES.register_module(force=True)
class DefaultFormatBundle3D:

    def __init__(
            self,
            class_names=None,
            classes=None,
            with_gt: bool = True,
            with_label: bool = True,
    ) -> None:
        self.class_names = class_names if class_names is not None else classes
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        # Format 3D data
        if "points" in results:
            if isinstance(results["points"], BasePoints):
                results["points"] = DC(results["points"].tensor)
            else:
                results["points"] = DC(to_tensor(results["points"]))

        for key in ["voxels", "coors", "voxel_centers", "num_points"]:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]), stack=False)

        if self.with_gt:
            if "gt_bboxes_3d_mask" in results:
                gt_bboxes_3d_mask = results["gt_bboxes_3d_mask"]
                results["gt_bboxes_3d"] = results["gt_bboxes_3d"][gt_bboxes_3d_mask]
                if "gt_names_3d" in results:
                    results["gt_names_3d"] = results["gt_names_3d"][gt_bboxes_3d_mask]
                if "centers2d" in results:
                    results["centers2d"] = results["centers2d"][gt_bboxes_3d_mask]
                if "depths" in results:
                    results["depths"] = results["depths"][gt_bboxes_3d_mask]

            if "gt_bboxes_mask" in results:
                gt_bboxes_mask = results["gt_bboxes_mask"]
                if "gt_bboxes" in results:
                    results["gt_bboxes"] = results["gt_bboxes"][gt_bboxes_mask]
                results["gt_names"] = results["gt_names"][gt_bboxes_mask]

            if self.with_label:
                if "gt_names" in results and len(results["gt_names"]) == 0:
                    results["gt_labels"] = np.array([], dtype=np.int64)
                    results["attr_labels"] = np.array([], dtype=np.int64)
                elif "gt_names" in results and isinstance(results["gt_names"][0], list):
                    if self.class_names is not None:
                        results["gt_labels"] = [
                            np.array([self.class_names.index(n) for n in res], dtype=np.int64)
                            for res in results["gt_names"]
                        ]
                elif "gt_names" in results:
                    if self.class_names is not None:
                        results["gt_labels"] = np.array(
                            [self.class_names.index(n) for n in results["gt_names"]],
                            dtype=np.int64,
                        )

                if "gt_names_3d" in results:
                    if self.class_names is not None:
                        results["gt_labels_3d"] = np.array(
                            [self.class_names.index(n) for n in results["gt_names_3d"]],
                            dtype=np.int64,
                        )

        if "img" in results:
            if isinstance(results["img"], list):
                results["img"] = DC(torch.stack(results["img"]), stack=True)
            else:
                results["img"] = DC(results["img"], stack=True)

        for key in [
            "proposals", "gt_bboxes", "gt_bboxes_ignore", "gt_labels",
            "gt_labels_3d", "attr_labels", "centers2d", "depths",
            "voxel_semantics", "mask_lidar", "mask_camera"
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))

        if "gt_bboxes_3d" in results:
            if isinstance(results["gt_bboxes_3d"], BaseInstance3DBoxes):
                results["gt_bboxes_3d"] = DC(results["gt_bboxes_3d"], cpu_only=True)
            else:
                results["gt_bboxes_3d"] = DC(to_tensor(results["gt_bboxes_3d"]))
        return results


@PIPELINES.register_module(force=True)
class Collect3D:
    """Collect data from the loader relevant to the specific task."""

    def __init__(
            self,
            keys,
            meta_keys=(
                    "camera_intrinsics", "camera2ego", "img_aug_matrix", "lidar_aug_matrix",
            ),
            meta_lis_keys=(
                    "filename", "timestamp", "ori_shape", "img_shape", "lidar2image",
                    "depth2img", "cam2img", "pad_shape", "scale_factor", "flip",
                    "pcd_horizontal_flip", "pcd_vertical_flip", "box_mode_3d",
                    "box_type_3d", "img_norm_cfg", "pcd_trans", "token",
                    "pcd_scale_factor", "pcd_rotation", "lidar_path",
                    "transformation_3d_flow", "occ_gt_path", "occ3d",
                    "surround_occ", "open_occ", "scene_token", "can_bus"
            ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_lis_keys = meta_lis_keys

    def __call__(self, results):
        data = {}
        for key in self.keys:
            if key not in self.meta_keys:
                data[key] = results[key]

        for key in self.meta_keys:
            if key in results:
                val = np.array(results[key])
                if isinstance(results[key], list):
                    data[key] = DC(to_tensor(val), stack=True)
                else:
                    data[key] = DC(to_tensor(val), stack=True, pad_dims=1)

        metas = {}
        for key in self.meta_lis_keys:
            if key in results:
                metas[key] = results[key]

        for possible_key in ['occ3d', 'surround_occ', 'open_occ', 'scene_token', 'can_bus']:
            if possible_key in results and possible_key not in metas:
                metas[possible_key] = results[possible_key]

        data["metas"] = DC(metas, cpu_only=True)
        return data


@PIPELINES.register_module(force=True)
class FormatBundle3D(DefaultFormatBundle3D):
    pass
