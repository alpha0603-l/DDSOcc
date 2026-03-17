import argparse
import copy
import os
import sys
import warnings
import inspect
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import path as osp

def add_cwd_to_path():
    cwd = os.getcwd()
    if cwd not in sys.path:
        print(f"Info: Adding Current Working Directory to PYTHONPATH: '{cwd}'")
        sys.path.insert(0, cwd)

add_cwd_to_path()
print("\n--- Python Search Paths (sys.path) ---")
print(f"Top priority path: {sys.path[0]}")
print("------------------------------------")
try:
    import mmdet3d
    print(f"Info: Successfully imported 'mmdet3d'.")
    print(f"Info: Location of imported 'mmdet3d': {mmdet3d.__file__}")
except ImportError as e:
    print(f"Fatal: Failed to import 'mmdet3d'. Ensure it is installed or in python path: {e}")
print("------------------------------------\n")
import mmcv

try:
    mmcv.__version__ = '2.1.0'
except Exception:
    pass

import mmengine
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import MODELS, DATASETS, TRANSFORMS, init_default_scope
from mmdet3d.structures import LiDARInstance3DBoxes, Det3DDataSample, CameraInstance3DBoxes

try:
    from mmdet3d.datasets import NuScenesDataset
except ImportError:
    try:
        from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
    except ImportError:
        print("Warning: Could not import NuScenesDataset. NuScenesDatasetOccupancy definition might fail.")

        class NuScenesDataset:
            def __init__(self, *args, **kwargs): pass

try:
    from torchpack.utils.config import configs
except ImportError:
    configs = None
    print("Warning: torchpack not found. Config loading might fail if it relies on torchpack specifics.")

def local_points_cam2img(points_3d, proj_mat, with_depth=False):
    points_num = list(points_3d.shape)[:-1]
    points_shape = np.concatenate([points_num, [1]], axis=0).tolist()
    assert len(proj_mat.shape) == 2, (
        f"The dimension of the projection matrix should be 2 instead of {len(proj_mat.shape)}."
    )
    d1, d2 = proj_mat.shape[:2]
    if d1 == 3 and d2 == 3:
        proj_mat_expanded = torch.eye(4, device=proj_mat.device, dtype=proj_mat.dtype)
        proj_mat_expanded[:3, :3] = proj_mat
        proj_mat = proj_mat_expanded
    elif d1 == 3 and d2 == 4:
        proj_mat_expanded = torch.eye(4, device=proj_mat.device, dtype=proj_mat.dtype)
        proj_mat_expanded[:3, :4] = proj_mat
        proj_mat = proj_mat_expanded
    if not isinstance(points_3d, torch.Tensor):
        points_3d = torch.tensor(points_3d)
    points_4 = torch.cat([points_3d, points_3d.new_ones(*points_shape)], dim=-1)
    point_2d = torch.matmul(points_4, proj_mat.t())
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    if with_depth:
        return torch.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)
    return point_2d_res

def plot_rect3d_on_img(img, num_rects, rect_corners, color=(0, 255, 0), thickness=1):
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))

    for i in range(num_rects):
        corners = rect_corners[i].astype(int)
        for start, end in line_indices:
            cv2.line(img, tuple(corners[start]), tuple(corners[end]), color, thickness, cv2.LINE_AA)
    return img.astype(np.uint8)


def draw_lidar_bbox3d_on_img(bboxes3d, raw_img, lidar2img_rt, img_metas, color=(0, 255, 0), thickness=1):
    img = raw_img.copy()

    corners_3d = bboxes3d.corners
    if isinstance(corners_3d, torch.Tensor):
        corners_3d = corners_3d.cpu().numpy()

    num_bbox = corners_3d.shape[0]

    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3),
         np.ones((num_bbox * 8, 1))], axis=-1)

    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()

    pts_2d = pts_4d @ lidar2img_rt.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)


def draw_camera_bbox3d_on_img(bboxes3d, raw_img, cam2img, img_metas, color=(0, 255, 0), thickness=1):
    img = raw_img.copy()
    corners_3d = bboxes3d.corners
    if isinstance(corners_3d, np.ndarray):
        corners_3d = torch.from_numpy(corners_3d)

    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)

    if not isinstance(cam2img, torch.Tensor):
        cam2img = torch.from_numpy(np.array(cam2img))

    cam2img = cam2img.float().cpu()
    points_3d_t = points_3d.float().cpu()

    uv_origin = local_points_cam2img(points_3d_t, cam2img)
    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)

try:
    import open3d as o3d
    from open3d import geometry

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Open3D not found. 3D visualization functions will utilize fallback or skip.")

def _draw_points(points, vis, points_size=2, point_color=(0.5, 0.5, 0.5), mode='xyz'):
    vis.get_render_option().point_size = points_size
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    points = points.copy()
    pcd = geometry.PointCloud()
    if mode == 'xyz':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = np.tile(np.array(point_color), (points.shape[0], 1))
    elif mode == 'xyzrgb':
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        points_colors = points[:, 3:6]
        if not ((points_colors >= 0.0) & (points_colors <= 1.0)).all():
            points_colors /= 255.0
    else:
        raise NotImplementedError

    pcd.colors = o3d.utility.Vector3dVector(points_colors)
    vis.add_geometry(pcd)
    return pcd, points_colors


def _draw_bboxes(bbox3d, vis, points_colors, pcd=None, bbox_color=(0, 1, 0),
                 points_in_box_color=(1, 0, 0), rot_axis=2, center_mode='lidar_bottom', mode='xyz'):
    if isinstance(bbox3d, torch.Tensor):
        bbox3d = bbox3d.cpu().numpy()
    bbox3d = bbox3d.copy()

    in_box_color = np.array(points_in_box_color)
    for i in range(len(bbox3d)):
        center = bbox3d[i, 0:3]
        dim = bbox3d[i, 3:6]
        yaw = np.zeros(3)
        yaw[rot_axis] = -bbox3d[i, 6]
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)

        if center_mode == 'lidar_bottom':
            center[rot_axis] += dim[rot_axis] / 2
        elif center_mode == 'camera_bottom':
            center[rot_axis] -= dim[rot_axis] / 2

        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)
        line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color(bbox_color)
        vis.add_geometry(line_set)

        if pcd is not None and mode == 'xyz':
            indices = box3d.get_point_indices_within_bounding_box(pcd.points)
            points_colors[indices] = in_box_color

    if pcd is not None:
        pcd.colors = o3d.utility.Vector3dVector(points_colors)
        vis.update_geometry(pcd)


class Visualizer(object):
    def __init__(self, points, bbox3d=None, save_path=None, points_size=2,
                 point_color=(0.5, 0.5, 0.5), bbox_color=(0, 1, 0),
                 points_in_box_color=(1, 0, 0), rot_axis=2, center_mode='lidar_bottom', mode='xyz'):
        super(Visualizer, self).__init__()
        if not HAS_OPEN3D:
            return

        self.o3d_visualizer = o3d.visualization.Visualizer()
        self.o3d_visualizer.create_window(visible=False)
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        self.o3d_visualizer.add_geometry(mesh_frame)

        self.points_size = points_size
        self.point_color = point_color
        self.bbox_color = bbox_color
        self.points_in_box_color = points_in_box_color
        self.rot_axis = rot_axis
        self.center_mode = center_mode
        self.mode = mode
        self.pcd = None
        self.points_colors = None

        if points is not None:
            self.pcd, self.points_colors = _draw_points(
                points, self.o3d_visualizer, points_size, point_color, mode)

        if bbox3d is not None:
            _draw_bboxes(bbox3d, self.o3d_visualizer, self.points_colors,
                         self.pcd, bbox_color, points_in_box_color, rot_axis,
                         center_mode, mode)

    def add_bboxes(self, bbox3d, bbox_color=None, points_in_box_color=None):
        if not HAS_OPEN3D: return
        if bbox_color is None: bbox_color = self.bbox_color
        if points_in_box_color is None: points_in_box_color = self.points_in_box_color
        _draw_bboxes(bbox3d, self.o3d_visualizer, self.points_colors, self.pcd,
                     bbox_color, points_in_box_color, self.rot_axis,
                     self.center_mode, self.mode)

    def show(self, save_path=None):
        if not HAS_OPEN3D: return
        self.o3d_visualizer.poll_events()
        self.o3d_visualizer.update_renderer()
        if save_path is not None:
            self.o3d_visualizer.capture_screen_image(save_path)
        self.o3d_visualizer.destroy_window()

def visualize_camera_adapter(path, image, bboxes=None, transform=None):
    if transform is None:
        return

    img_with_box = image.copy()

    if bboxes is not None and len(bboxes) > 0:
        if isinstance(bboxes, LiDARInstance3DBoxes):
            img_with_box = draw_lidar_bbox3d_on_img(
                bboxes, image, transform, img_metas={}, color=(0, 255, 0)
            )
        elif isinstance(bboxes, CameraInstance3DBoxes):
            img_with_box = draw_camera_bbox3d_on_img(
                bboxes, image, transform, img_metas={}, color=(0, 255, 0)
            )

    mmengine.mkdir_or_exist(os.path.dirname(path))
    mmcv.imwrite(img_with_box, path)


def visualize_lidar_adapter(path, points, bboxes=None, xlim=None, ylim=None):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_aspect('equal')

    points_draw = points
    if xlim and ylim:
        mask = (points[:, 0] > xlim[0]) & (points[:, 0] < xlim[1]) & \
               (points[:, 1] > ylim[0]) & (points[:, 1] < ylim[1])
        points_draw = points[mask]

    ax.scatter(points_draw[:, 0], points_draw[:, 1], s=0.5, c='gray', alpha=0.5)

    if bboxes is not None:
        corners = bboxes.corners
        if isinstance(corners, torch.Tensor):
            corners = corners.cpu().numpy()

        bev_corners = corners[:, [0, 1, 2, 3], :2]
        for i in range(len(bev_corners)):
            poly = bev_corners[i]
            poly = np.vstack([poly, poly[0]])
            ax.plot(poly[:, 0], poly[:, 1], c='red', linewidth=1)

    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
    plt.axis('off')
    plt.tight_layout()

    mmengine.mkdir_or_exist(os.path.dirname(path))
    plt.savefig(path, dpi=100)
    plt.close(fig)

@DATASETS.register_module(name='NuScenesDatasetOccupancy', force=True)
class NuScenesDatasetOccupancy(NuScenesDataset):

    def __init__(
            self,
            ann_file,
            pipeline=None,
            dataset_root=None,
            object_classes=None,
            map_classes=None,
            load_interval=1,
            with_velocity=True,
            modality=None,
            box_type_3d="LiDAR",
            filter_empty_gt=True,
            test_mode=False,
            eval_version="detection_cvpr_2019",
            use_valid_flag=False,
            resample=True,
            data_type='occ3d',
            **kwargs
    ) -> None:
        if dataset_root is not None:
            kwargs['data_root'] = dataset_root

        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            dataset_root=dataset_root,
            object_classes=object_classes,
            map_classes=map_classes,
            load_interval=load_interval,
            with_velocity=with_velocity,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            eval_version=eval_version,
            use_valid_flag=use_valid_flag,
            **kwargs
        )
        self.resample = resample
        assert data_type in ['occ3d', 'surround_occ', 'open_occ']
        self.data_type = data_type

    def get_data_info(self, index):
        input_dict = super(NuScenesDatasetOccupancy, self).get_data_info(index)
        if not hasattr(self, 'data_infos') and hasattr(self, 'data_list'):
            info = self.data_list[index]
        else:
            info = self.data_infos[index]
        if self.data_type not in input_dict:
            input_dict[self.data_type] = {}
        if self.data_type in info:
            input_dict[self.data_type]['occ_gt_path'] = info[self.data_type]['occ_path']

        return input_dict

    def evaluate(self, results, **kwargs):

        return {}


def recursive_eval(obj, globals=None):
    if globals is None: globals = copy.deepcopy(obj)
    if isinstance(obj, dict):
        for key in obj: obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj): obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        try:
            obj = eval(obj[2:-1], globals)
            obj = recursive_eval(obj, globals)
        except Exception:
            pass
    return obj


def clean_pipeline_config(pipeline):
    if not isinstance(pipeline, list): return pipeline
    cleaned_pipeline = []
    for transform_cfg in pipeline:
        if not isinstance(transform_cfg, dict):
            cleaned_pipeline.append(transform_cfg)
            continue
        t_type = transform_cfg.get('type')
        if not t_type:
            cleaned_pipeline.append(transform_cfg)
            continue
        target_cls = TRANSFORMS.get(t_type)
        if not target_cls:
            cleaned_pipeline.append(transform_cfg)
            continue
        try:
            sig = inspect.signature(target_cls.__init__)
            valid_params = {p for p in sig.parameters if p != 'self'}
        except (TypeError, ValueError):
            cleaned_pipeline.append(transform_cfg)
            continue

        original_keys = list(transform_cfg.keys())
        for key in original_keys:
            if key not in valid_params and key != 'type':
                transform_cfg.pop(key)

        if 'transforms' in transform_cfg:
            transform_cfg['transforms'] = clean_pipeline_config(transform_cfg['transforms'])

        cleaned_pipeline.append(transform_cfg)
    return cleaned_pipeline

def main() -> None:
    init_default_scope('mmdet3d')

    parser = argparse.ArgumentParser(description="Visualize MMDet3D Results")
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--mode", type=str, default="pred", choices=["gt", "pred"], help="Visualize GT or Prediction")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint file")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Data split to use")
    parser.add_argument("--out-dir", type=str, default="viz", help="Output directory for visualization")
    parser.add_argument("--interval", type=int, default=1, help="Visualization interval")

    args, opts = parser.parse_known_args()

    if configs is not None:
        configs.load(args.config, recursive=True)
        configs.update(opts)
        cfg_dict = recursive_eval(configs)
        cfg = Config(cfg_dict, filename=args.config)
    else:
        cfg = Config.fromfile(args.config)

    if cfg.get('custom_imports', None):
        print("Info: Handling custom imports...")
        from mmengine.utils import import_modules_from_strings
        custom_imports = cfg['custom_imports']
        if isinstance(custom_imports, dict):
            import_modules_from_strings(**custom_imports)
        elif isinstance(custom_imports, list):
            import_modules_from_strings(imports=custom_imports)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    split_key = args.split
    if 'data' in cfg and split_key in cfg.data:
        split_cfg = cfg.data[split_key]
    else:
        if 'test_dataloader' in cfg and split_key == 'val':
            split_cfg = cfg.test_dataloader.dataset
        elif 'val_dataloader' in cfg and split_key == 'val':
            split_cfg = cfg.val_dataloader.dataset
        else:
            raise KeyError(f"Cannot find key '{split_key}' in the config's data section.")

    if isinstance(split_cfg, dict) or isinstance(split_cfg, Config):
        if 'pipeline' in split_cfg:
            split_cfg['pipeline'] = clean_pipeline_config(split_cfg['pipeline'])
        elif 'dataset' in split_cfg and 'pipeline' in split_cfg['dataset']:
            split_cfg['dataset']['pipeline'] = clean_pipeline_config(split_cfg['dataset']['pipeline'])

    print(f"--- Building dataset with raw config (pipeline cleaned) ---")
    dataset = DATASETS.build(split_cfg)

    dataflow = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=cfg.get('data', {}).get('workers_per_gpu', 1),
        shuffle=False,
        collate_fn=pseudo_collate,
        pin_memory=True
    )

    print("--- Building model ---")
    model = MODELS.build(cfg.model)
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded manually to be compatible with PyTorch 2.6+.")

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    frame_id = -1
    print(f"--- Starting visualization loop ---")
    mmengine.mkdir_or_exist(args.out_dir)

    for data in tqdm(dataflow):
        frame_id += 1
        if frame_id % args.interval != 0: continue

        if isinstance(data, dict) and 'data_samples' in data:
            data_sample = data['data_samples'][0]
            metas = data_sample.metainfo
        elif isinstance(data, dict) and 'img_metas' in data:
            metas = data['img_metas'][0].data[0]
            data_sample = None
        else:
            if isinstance(data.get('data_samples'), list):
                data_sample = data['data_samples'][0]
                metas = data_sample.metainfo
            else:
                continue

        token = metas.get("token", "")
        ts = metas.get("timestamp", frame_id)
        name = f"{ts}-{token}"

        outputs = None
        if args.mode == "pred":
            with torch.no_grad():
                outputs = model.test_step(data)

        bboxes = None
        if args.mode == "gt":
            if data_sample and hasattr(data_sample, 'gt_instances_3d'):
                bboxes = data_sample.gt_instances_3d.bboxes_3d
        elif args.mode == "pred" and outputs is not None:
            if 'pred_instances_3d' in outputs[0]:
                bboxes = outputs[0].pred_instances_3d.bboxes_3d

        if 'inputs' in data and 'img' in data['inputs']:
            imgs = data['inputs']['img']
            if isinstance(imgs, torch.Tensor):
                if imgs.dim() == 5:
                    imgs = imgs[0]

                for k in range(imgs.shape[0]):
                    image_tensor = imgs[k]
                    image = image_tensor.permute(1, 2, 0).cpu().numpy()
                    image = (image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
                    image = np.clip(image, 0, 255).astype(np.uint8)

                    viz_path = os.path.join(args.out_dir, f"camera-{k}", f"{name}.png")
                    mmengine.mkdir_or_exist(os.path.dirname(viz_path))

                    cv2.imwrite(viz_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            elif isinstance(imgs, list):
                for k, image_tensor in enumerate(imgs):
                    image = image_tensor.permute(1, 2, 0).cpu().numpy()
                    image = (image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
                    image = np.clip(image, 0, 255).astype(np.uint8)
                    viz_path = os.path.join(args.out_dir, f"camera-{k}", f"{name}.png")
                    mmengine.mkdir_or_exist(os.path.dirname(viz_path))
                    cv2.imwrite(viz_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
