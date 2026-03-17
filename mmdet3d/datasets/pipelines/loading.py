import os
from typing import Any, Dict, Tuple

import torch
import mmcv
import numpy as np
from PIL import Image
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.map_api import locations as LOCATIONS
from mmengine.registry import TRANSFORMS

try:
    from mmdet3d.datasets.pipelines.loading_utils import load_augmented_point_cloud, reduce_LiDAR_beams
except ImportError:
    try:
        from .loading_utils import load_augmented_point_cloud, reduce_LiDAR_beams
    except ImportError:
        raise

try:
    from mmdet3d.structures.points import BasePoints, get_points_type
except ImportError:
    from mmdet3d.core.points import BasePoints, get_points_type

try:
    from mmdet.datasets.transforms import LoadAnnotations
except ImportError:
    class LoadAnnotations:
        def __init__(self, with_bbox=True, with_label=True, with_mask=False, with_seg=False, poly2mask=True):
            self.with_bbox = with_bbox
            self.with_label = with_label
            self.with_mask = with_mask
            self.with_seg = with_seg
            self.poly2mask = poly2mask

        def __call__(self, results):
            return results


@TRANSFORMS.register_module(force=True)
class LoadMultiViewImageFromFiles:
    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type
        print(f">>> [Registry] LoadMultiViewImageFromFiles (BGR Fix) 已注册")

    def __call__(self, results):
        filename = results["image_paths"]
        images = []
        for name in filename:
            img = mmcv.imread(name, self.color_type)
            if self.to_float32:
                img = img.astype(np.float32)
            images.append(img)
        results["filename"] = filename
        results["img"] = images
        results["img_shape"] = images[0].shape[:2]
        results["ori_shape"] = images[0].shape[:2]
        results["pad_shape"] = images[0].shape[:2]
        results["scale_factor"] = 1.0
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@TRANSFORMS.register_module(force=True)
class LoadMultiViewImageFromFilesWaymo:

    def __init__(self, to_float32=False, color_type="unchanged", image_type=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.image_type = image_type
        assert image_type in ['jpg', 'png', None]

    def __call__(self, results):
        filename = results["image_paths"]
        images = []
        img_shapes = []
        ori_shapes = []
        pad_shapes = []

        for name in filename:
            if self.image_type:
                cur_name, cur_ext = os.path.splitext(name)
                name = cur_name + '.' + self.image_type
            image = Image.open(name)
            images.append(image)
            img_shapes.append(image.size)
            ori_shapes.append(image.size)
            pad_shapes.append(image.size)

        results["filename"] = filename
        results["img"] = images
        results["img_shape"] = img_shapes
        results["ori_shape"] = ori_shapes
        results["pad_shape"] = pad_shapes
        results["scale_factor"] = 1.0

        return results


@TRANSFORMS.register_module(force=True)
class LoadPointsFromMultiSweeps:

    def __init__(
            self,
            sweeps_num=10,
            load_dim=5,
            use_dim=[0, 1, 2, 4],
            pad_empty_sweeps=False,
            remove_close=False,
            test_mode=False,
            load_augmented=None,
            reduce_beams=None,
    ):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        points = results["points"]
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results["timestamp"] / 1e6
        if self.pad_empty_sweeps and len(results["sweeps"]) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results["sweeps"]) <= self.sweeps_num:
                choices = np.arange(len(results["sweeps"]))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                if not self.load_augmented:
                    choices = np.random.choice(
                        len(results["sweeps"]), self.sweeps_num, replace=False
                    )
                else:
                    choices = np.random.choice(
                        len(results["sweeps"]) - 1, self.sweeps_num, replace=False
                    )
            for idx in choices:
                sweep = results["sweeps"][idx]
                points_sweep = self._load_points(sweep["data_path"])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                if self.reduce_beams and self.reduce_beams < 32:
                    points_sweep = reduce_LiDAR_beams(points_sweep, self.reduce_beams)

                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep["timestamp"] / 1e6
                points_sweep[:, :3] = (
                        points_sweep[:, :3] @ sweep["sensor2lidar_rotation"].T
                )
                points_sweep[:, :3] += sweep["sensor2lidar_translation"]
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results["points"] = points
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(sweeps_num={self.sweeps_num})"


@TRANSFORMS.register_module(force=True)
class LoadBEVSegmentation:
    def __init__(
            self,
            dataset_root: str,
            xbound: Tuple[float, float, float],
            ybound: Tuple[float, float, float],
            classes: Tuple[str, ...],
    ) -> None:
        super().__init__()
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.classes = classes

        self.maps = {}
        for location in LOCATIONS:
            self.maps[location] = NuScenesMap(dataset_root, location)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        lidar2point = data["lidar_aug_matrix"]
        point2lidar = np.linalg.inv(lidar2point)
        lidar2ego = data["lidar2ego"]
        ego2global = data["ego2global"]
        lidar2global = ego2global @ lidar2ego @ point2lidar

        map_pose = lidar2global[:2, 3]
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])

        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
        patch_angle = yaw / np.pi * 180

        mappings = {}
        for name in self.classes:
            if name == "drivable_area*":
                mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                mappings[name] = ["road_divider", "lane_divider"]
            else:
                mappings[name] = [name]

        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))

        location = data["location"]
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        )
        masks = masks.transpose(0, 2, 1)
        masks = masks.astype(bool)

        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.int64)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1

        data["gt_masks_bev"] = labels
        return data


@TRANSFORMS.register_module(force=True)
class LoadPointsFromFile:

    def __init__(
            self,
            coord_type,
            load_dim=6,
            use_dim=[0, 1, 2],
            shift_height=False,
            use_color=False,
            load_augmented=None,
            reduce_beams=None,
            tanh_dim=None,
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
                max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams
        self.tanh_dim = tanh_dim

    def _load_points(self, lidar_path):
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)

        return points

    def __call__(self, results):
        lidar_path = results["lidar_path"]
        points = self._load_points(lidar_path)
        points = points.reshape(-1, self.load_dim)
        if self.reduce_beams and self.reduce_beams < 32:
            points = reduce_LiDAR_beams(points, self.reduce_beams)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        if self.tanh_dim:
            points[:, self.tanh_dim] = np.tanh(points[:, self.tanh_dim])

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims
        )
        results["points"] = points

        return results


@TRANSFORMS.register_module(force=True)
class LoadAnnotations3D(LoadAnnotations):

    def __init__(
            self,
            with_bbox_3d=True,
            with_label_3d=True,
            with_attr_label=False,
            with_bbox=False,
            with_label=False,
            with_mask=False,
            with_seg=False,
            with_bbox_depth=False,
            poly2mask=True,
    ):
        super().__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            poly2mask=poly2mask,
        )
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label

    def _load_bboxes_3d(self, results):
        results["gt_bboxes_3d"] = results["ann_info"]["gt_bboxes_3d"]
        results["bbox3d_fields"].append("gt_bboxes_3d")
        return results

    def _load_bboxes_depth(self, results):
        results["centers2d"] = results["ann_info"]["centers2d"]
        results["depths"] = results["ann_info"]["depths"]
        return results

    def _load_labels_3d(self, results):
        results["gt_labels_3d"] = results["ann_info"]["gt_labels_3d"]
        return results

    def _load_attr_labels(self, results):
        results["attr_labels"] = results["ann_info"]["attr_labels"]
        return results

    def __call__(self, results):
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)

        return results


@TRANSFORMS.register_module(force=True)
class LoadOccGTFromFile:

    def __init__(self, data_type='occ3d'):
        assert data_type in ['occ3d', 'surround_occ', 'open_occ']
        self.data_type = data_type
        self.occ_shape = {'surround_occ': [200, 200, 16], 'open_occ': [512, 512, 40]}

    def __call__(self, results):
        if self.data_type == 'occ3d':
            occ_gt_path = results['occ3d']['occ_gt_path']
            if not occ_gt_path.endswith('labels.npz'):
                occ_gt_path = os.path.join(occ_gt_path, "labels.npz")

            occ_labels = np.load(occ_gt_path)
            semantics = occ_labels['semantics']
            mask_lidar = occ_labels['mask_lidar']
            mask_camera = occ_labels['mask_camera']

            results['voxel_semantics'] = semantics  # (200, 200, 16)
            results['mask_lidar'] = mask_lidar  # (200, 200, 16)
            results['mask_camera'] = mask_camera  # (200, 200, 16)
        else:
            occ = np.load(results[self.data_type]['occ_gt_path'])
            if self.data_type == 'open_occ':
                occ = occ[..., [2, 1, 0, 3]]
            occ = occ.astype(np.float32)

            gt = np.zeros(self.occ_shape[self.data_type], dtype=np.float32)
            occ[..., 3][occ[..., 3] == 0] = 255
            coords = occ[:, :3].astype(np.int32)
            gt[coords[:, 0], coords[:, 1], coords[:, 2]] = occ[:, 3]

            results['voxel_semantics'] = gt.astype(np.int32)
            results['mask_lidar'] = np.zeros_like(gt) - 1
            results['mask_camera'] = np.zeros_like(gt) - 1

        results["occ_aug_matrix"] = np.eye(4).astype(np.float32)
        return results


@TRANSFORMS.register_module(force=True)
class LoadOccGTFromFileWaymo(object):

    def __init__(
            self,
            data_root,
            use_larger=True,
            crop_x=False,
            num_classes=16,
            free_label=23,
            use_infov=False,
    ):
        self.use_larger = use_larger
        self.data_root = data_root
        self.crop_x = crop_x
        self.num_classes = num_classes
        self.free_label = free_label
        self.use_infov = use_infov

    def __call__(self, results):
        pts_filename = results['pts_filename']
        basename = os.path.basename(pts_filename)
        seq_name = basename[1:4]
        frame_name = basename[4:7]
        if self.use_larger:
            file_path = os.path.join(self.data_root, seq_name, '{}_04.npz'.format(frame_name))
        else:
            file_path = os.path.join(self.data_root, seq_name, '{}.npz'.format(frame_name))
        occ_labels = np.load(file_path)
        semantics = occ_labels['voxel_label']
        mask_infov = occ_labels['infov']
        mask_lidar = occ_labels['origin_voxel_state']
        mask_camera = occ_labels['final_voxel_state']
        if self.crop_x:
            w, h, d = semantics.shape
            semantics = semantics[w // 2:, :, :]
            mask_infov = mask_infov[w // 2:, :, :]
            mask_lidar = mask_lidar[w // 2:, :, :]
            mask_camera = mask_camera[w // 2:, :, :]

        semantics[semantics == self.free_label] = self.num_classes - 1
        results['voxel_semantics'] = semantics
        results['mask_infov'] = mask_infov
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera
        if self.use_infov:
            results['mask_camera'] = np.logical_and(mask_infov, mask_camera)

        results["occ_aug_matrix"] = np.eye(4).astype(np.float32)
        return results

    def __repr__(self):
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)