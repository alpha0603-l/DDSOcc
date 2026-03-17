import argparse
import os
import sys
import time
import re
import logging
import warnings
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import multiprocessing as mp
try:
    if mp.get_start_method(allow_none=True) != "fork":
        mp.set_start_method("fork", force=True)
except RuntimeError:
    pass
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.getcwd())
_original_inverse = torch.inverse
def patched_inverse(input, *args, **kwargs):
    try:
        return _original_inverse(input, *args, **kwargs)
    except RuntimeError as e:
        if 'CUSOLVER' in str(e) or 'MAGMA' in str(e) or 'INTERNAL_ERROR' in str(e):
            return _original_inverse(input.cpu().float()).to(device=input.device, dtype=input.dtype)
        raise e
torch.inverse = patched_inverse
_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load
import mmengine
from mmengine.config import Config
from mmengine.registry import MODELS, DATASETS
from mmengine.dist import get_dist_info, init_dist, all_gather_object
from mmengine.runner import load_checkpoint
from mmengine.utils import ProgressBar
from mmengine.logging import MMLogger
from torchpack.utils.config import configs
from torch.utils.data import DataLoader, DistributedSampler
from mmengine.dataset import pseudo_collate
try:
    from mmdet3d.structures import LiDARInstance3DBoxes
except ImportError:
    LiDARInstance3DBoxes = None
try:
    from mmdet3d.models.fusion_models.bevfusion import BEVFusion
    MODELS.register_module(module=BEVFusion, name='BEVFusion', force=True)
    import mmdet3d.datasets.dataset_wrappers
    import mmdet3d.datasets.nuscenes_occupancy_dataset
except ImportError:
    pass

def cast_tensor_to_float32(obj):
    if isinstance(obj, torch.Tensor):
        if obj.dtype in (torch.float16, torch.bfloat16):
            return obj.float()
        return obj
    if isinstance(obj, list):
        return [cast_tensor_to_float32(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(cast_tensor_to_float32(x) for x in obj)
    if isinstance(obj, dict):
        return {k: cast_tensor_to_float32(v) for k, v in obj.items()}
    return obj

def recursive_to_cpu(data):
    if isinstance(data, torch.Tensor):
        return data.cpu()
    if hasattr(data, 'to') and hasattr(data, 'device'):
        return data.to('cpu')
    if isinstance(data, dict):
        return {k: recursive_to_cpu(v) for k, v in data.items()}
    if isinstance(data, list):
        return [recursive_to_cpu(v) for v in data]
    if isinstance(data, tuple):
        return tuple(recursive_to_cpu(v) for v in data)
    return data

def ensure_valid_boxes(res: dict):
    if 'pts_bbox' not in res:
        if 'boxes_3d' in res:
            res['pts_bbox'] = {
                'boxes_3d': res.get('boxes_3d'),
                'scores_3d': res.get('scores_3d'),
                'labels_3d': res.get('labels_3d')
            }
        else:
            res['pts_bbox'] = {}

    bbox_res = res['pts_bbox']

    invalid = False
    if 'boxes_3d' not in bbox_res or bbox_res['boxes_3d'] is None:
        invalid = True

    if invalid:
        if LiDARInstance3DBoxes is not None:
            empty_tensor = torch.empty((0, 9), dtype=torch.float32)
            bbox_res['boxes_3d'] = LiDARInstance3DBoxes(empty_tensor, box_dim=9)
            bbox_res['scores_3d'] = torch.empty((0,), dtype=torch.float32)
            bbox_res['labels_3d'] = torch.empty((0,), dtype=torch.long)
        else:
            bbox_res['boxes_3d'] = torch.empty((0, 9), dtype=torch.float32)
            bbox_res['scores_3d'] = torch.empty((0,), dtype=torch.float32)
            bbox_res['labels_3d'] = torch.empty((0,), dtype=torch.long)

    return res

def move_to_cuda_fp32(obj):
    if isinstance(obj, torch.Tensor):
        t = obj.cuda(non_blocking=True)
        if t.dtype in (torch.float16, torch.bfloat16):
            t = t.float()
        return t
    if isinstance(obj, dict):
        return {k: move_to_cuda_fp32(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_cuda_fp32(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_cuda_fp32(v) for v in obj)
    return obj

def auto_fix_config(cfg, args=None):
    NOT_FOUND = object()

    def get_var_by_path(path, root_dict):
        try:
            if path in root_dict:
                return root_dict[path]
            parts = re.split(r'\.|\[|\]', path)
            parts = [p for p in parts if p]
            current = root_dict
            for p in parts:
                if isinstance(current, dict):
                    if p in current:
                        current = current[p]
                    else:
                        return NOT_FOUND
                elif isinstance(current, (list, tuple)):
                    if p.isdigit() and int(p) < len(current):
                        current = current[int(p)]
                    else:
                        return NOT_FOUND
                else:
                    return NOT_FOUND
            return current
        except Exception:
            return NOT_FOUND

    def fix_value(text, location_info=""):
        if not isinstance(text, str) or '${' not in text:
            return text
        pattern = re.compile(r'\$\{(.+?)\}')
        while True:
            match = pattern.search(text)
            if not match:
                break
            full_str, content = match.group(0), match.group(1).strip()
            resolved = None

            val = get_var_by_path(content, cfg._cfg_dict)
            if val is not NOT_FOUND:
                if full_str == text:
                    return val
                if isinstance(val, (str, int, float, bool)) or val is None:
                    resolved = str(val)
            elif '+' in content:
                parts = content.split('+')
                temp = ""
                valid = True
                for p in parts:
                    p = p.strip()
                    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
                        temp += p[1:-1]
                    else:
                        v = get_var_by_path(p, cfg._cfg_dict)
                        if v is not NOT_FOUND and isinstance(v, str):
                            temp += v
                        else:
                            valid = False
                            break
                if valid:
                    resolved = temp

            if resolved is not None:
                text = text.replace(full_str, resolved)
            else:
                raise ValueError(f"Cannot resolve config variable: {content}")
        return text

    def recursive_replace(obj, parent_key=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                current_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, str):
                    obj[k] = fix_value(v, location_info=current_key)
                elif isinstance(v, (list, dict)):
                    recursive_replace(v, parent_key=current_key)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                current_key = f"{parent_key}[{i}]"
                if isinstance(item, str):
                    obj[i] = fix_value(item, location_info=current_key)
                elif isinstance(item, (list, dict)):
                    recursive_replace(item, parent_key=current_key)

    recursive_replace(cfg._cfg_dict)
    return cfg

def parse_args():
    parser = argparse.ArgumentParser(description='MMEngine Dist Test (FAST FP32 + FORK workers)')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--eval', type=str, nargs='+', default=['bbox', 'mIoU'])
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--show-dir', help='directory where painted images will be saved')
    parser.add_argument('--use-test-set', action='store_true')
    parser.add_argument('--format-only', action='store_true')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--prefetch-factor', type=int, default=2)

    args, opts = parser.parse_known_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args, opts

def main():
    TEST_SEED = 777
    import random
    import numpy as np
    from scipy.spatial.transform import Rotation as R_tool
    
    random.seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    torch.manual_seed(TEST_SEED)
    torch.cuda.manual_seed_all(TEST_SEED)
    torch.backends.cudnn.deterministic = False 
    
    print(f"\n>>> [Init] Random Seed HARDCODED to {TEST_SEED} for Fair Robustness Comparison! <<<\n")
    args, opts = parse_args()

    if args.launcher == 'none':
        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
            args.launcher = 'pytorch'
            if 'LOCAL_RANK' in os.environ:
                args.local_rank = int(os.environ['LOCAL_RANK'])

    distributed = (args.launcher != 'none')
    if distributed:
        init_dist(args.launcher, backend='nccl')

    rank, world_size = get_dist_info()

    if rank == 0:
        print(">>> [Info] Loading Config...")
        if distributed:
            print(f">>> [Info] Distributed Mode ENABLED. World Size: {world_size}")
        else:
            print(">>> [Warning] Distributed Mode DISABLED. Running on SINGLE card?")
        print(">>> [Info] Precision Mode: FP32 (TF32 Disabled).")
        print(f">>> [Info] DataLoader: num_workers={args.num_workers}, persistent_workers=True, prefetch_factor={args.prefetch_factor}, mp_context=fork")
        sys.stdout.flush()

    configs.load(args.config, recursive=True)
    if opts:
        configs.update(opts)
    cfg = Config(configs)

    cfg = auto_fix_config(cfg, args)

    TEST_DROP_ALL  = False
    TEST_DROP_PROB = 0.0
    TEST_IS_NOISE  = False

    failure_cfg = dict(
        type='RobustCameraDropout',
        drop_prob=TEST_DROP_PROB,
        drop_all=TEST_DROP_ALL,
        black_out=not TEST_IS_NOISE
    )

    if hasattr(cfg, 'test_pipeline'):
        pipeline = cfg.test_pipeline
    elif 'test_pipeline' in cfg.data.test:
        pipeline = cfg.data.test.pipeline
    elif 'pipeline' in cfg.data.test:
        pipeline = cfg.data.test.pipeline
    else:
        pipeline = cfg.data.val.pipeline

    new_pipeline = []
    inserted = False
    for step in pipeline:
        new_pipeline.append(step)
        if step['type'] == 'LoadMultiViewImageFromFiles':
            new_pipeline.append(failure_cfg)
            inserted = True
            
    if not inserted:
        new_pipeline.insert(0, failure_cfg)

    if hasattr(cfg, 'test_pipeline'): cfg.test_pipeline = new_pipeline
    if 'test_pipeline' in cfg.data.test: cfg.data.test.pipeline = new_pipeline
    if 'pipeline' in cfg.data.test: cfg.data.test.pipeline = new_pipeline
    if 'val' in cfg.data and 'pipeline' in cfg.data.val: cfg.data.val.pipeline = new_pipeline

    if args.use_test_set:
        test_dataset_cfg = cfg.data.test
    else:
        if cfg.get('data') and cfg.data.get('val'):
            test_dataset_cfg = cfg.data.val
        elif cfg.get('val_dataloader'):
            test_dataset_cfg = cfg.val_dataloader.dataset
        else:
            test_dataset_cfg = cfg.data.test

    test_dataset_cfg['test_mode'] = True

    if rank == 0:
        print(f">>> [Info] Building Dataset: {test_dataset_cfg['type']}")
        sys.stdout.flush()

    dataset = DATASETS.build(test_dataset_cfg)

    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    if sampler is not None:
        sampler.set_epoch(0)

    dl_kwargs = dict(
        dataset=dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=pseudo_collate,      
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        multiprocessing_context="fork",   
    )
    if args.num_workers > 0:
        dl_kwargs["prefetch_factor"] = args.prefetch_factor

    data_loader = DataLoader(**dl_kwargs)

    cfg.model.train_cfg = None
    if rank == 0:
        mm_logger = MMLogger.get_current_instance()
        original_level = mm_logger.level
        mm_logger.setLevel(logging.WARNING)

    model = MODELS.build(cfg.model)

    if rank == 0:
        mm_logger.setLevel(original_level)

    if rank == 0:
        print(f">>> [Info] Loading checkpoint from {args.checkpoint}...")
        sys.stdout.flush()

    try:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    except Exception:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)

    if hasattr(dataset, 'CLASSES'):
        model.CLASSES = dataset.CLASSES

    model.cuda()
    model.eval()

    results = []

    if rank == 0:
        print(f">>> [Info] Start Inference ({len(dataset)} samples)...")
        print(">>> [Info] Force FP32 forward (no autocast).")
        prog_bar = ProgressBar(len(data_loader))
        sys.stdout.flush()



    TEST_DROP_LIDAR_ALL  = False
    TEST_DROP_LIDAR_RATE = False
    KEEP_RATIO           = 1.0

    results = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            data = move_to_cuda_fp32(data)

            if 'points' in data:
                new_points = []
                
                if TEST_DROP_LIDAR_ALL:
                    for p in data['points']:
                        p_tensor = p.data if hasattr(p, 'data') else p
                        new_points.append(p_tensor * 0)
                    data['points'] = new_points
                    
                elif TEST_DROP_LIDAR_RATE:
                    for p in data['points']:
                        p_tensor = p.data if hasattr(p, 'data') else p
                        num_points = p_tensor.shape[0]
                        keep_num = max(1, int(num_points * KEEP_RATIO))
                        indices = torch.randperm(num_points, device=p_tensor.device)[:keep_num]
                        new_points.append(p_tensor[indices])
                    data['points'] = new_points
            out = model(return_loss=False, rescale=True, **data)
            out = cast_tensor_to_float32(out)
            results.extend(out)
        if rank == 0:
            prog_bar.update()

    if distributed:
        if rank == 0:
            print(f"\n>>> [Info] Gathering results from {world_size} GPUs...")
            sys.stdout.flush()

        collected = all_gather_object(results)
        final_results = []
        if rank == 0:
            max_len = max(len(r) for r in collected)
            for i in range(max_len):
                for gpu_rank in range(world_size):
                    if i < len(collected[gpu_rank]):
                        final_results.append(collected[gpu_rank][i])
            if len(final_results) > len(dataset):
                final_results = final_results[:len(dataset)]
    else:
        final_results = results

    if rank == 0:
        print(f"\n>>> [Info] Post-processing results and SANITIZING Boxes...")
        sys.stdout.flush()

        sanitized = []
        none_count = 0
        for res in final_results:
            cpu_res = recursive_to_cpu(res)
            valid = ensure_valid_boxes(cpu_res)

            try:
                b = valid.get('pts_bbox', {}).get('boxes_3d', None)
                if hasattr(b, 'tensor') and b.tensor.numel() == 0:
                    none_count += 1
            except Exception:
                pass

            sanitized.append(valid)

        final_results = sanitized

        if none_count > 0:
            print(f">>> [Warning] Detected {none_count} samples with NO detections (replaced None with empty boxes).")
        if args.out:
            mmengine.dump(final_results, args.out)
            print(f">>> [Info] Saved results to: {args.out}")
        eval_cfg = cfg.get("evaluation")
        eval_kwargs = {} if eval_cfg is None else eval_cfg.copy()
        for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule", "pipeline"]:
            eval_kwargs.pop(key, None)
        if args.eval:
            eval_kwargs.update(dict(metric=args.eval))
        if args.show_dir:
            eval_kwargs['show_dir'] = args.show_dir

        print(">>> [Info] Starting Evaluation...")
        sys.stdout.flush()

        try:
            if hasattr(dataset, 'format_results') and args.format_only:
                dataset.format_results(final_results, **eval_kwargs)
            else:
                metrics = dataset.evaluate(final_results, **eval_kwargs)
                print("\n" + "=" * 20 + "  Metrics " + "=" * 20)
                for k, v in metrics.items():
                    print(f"{k}: {v}")
                print("=" * 50 + "\n")
        except Exception as e:
            print(f">>> [Error] Evaluation Failed: {e}")
            import traceback
            traceback.print_exc()

    # Clean shutdown (avoid NCCL leak warning)
    if distributed:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == '__main__':
    main()
