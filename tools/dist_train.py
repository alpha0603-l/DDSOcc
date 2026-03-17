import argparse
import os
import sys
import time
import re
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import warnings
import mmengine
import logging
import copy
import types
from torchpack.utils.config import configs
from collections.abc import Mapping, Sequence
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate
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

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import MODELS, DATASETS, HOOKS
from mmengine.dist import get_dist_info, init_dist, all_gather_object
from mmengine.hooks import Hook
from mmengine.utils import ProgressBar
from mmengine.logging import MMLogger

def naive_collate(batch):
    if not batch: return {}
    elem = batch[0]
    elem_type = type(elem)
    if 'DataContainer' in elem_type.__name__ or 'DC' in elem_type.__name__:
        stacked = getattr(elem, 'stack', True)
        if stacked:
            return default_collate([b.data for b in batch])
        else:
            return [b.data for b in batch]
    if isinstance(elem, Mapping):
        return {key: naive_collate([d[key] for d in batch]) for key in elem}
    if isinstance(elem, Sequence) and not isinstance(elem, (str, bytes)):
        return [naive_collate(samples) for samples in zip(*batch)]
    return default_collate(batch)

def cast_tensor_to_float32(obj):
    if isinstance(obj, torch.Tensor):
        if obj.dtype in [torch.bfloat16, torch.float16]:
            return obj.float()
        return obj
    elif isinstance(obj, list):
        return [cast_tensor_to_float32(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(cast_tensor_to_float32(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: cast_tensor_to_float32(v) for k, v in obj.items()}
    return obj

def recursive_to_cpu(data):
    if isinstance(data, torch.Tensor):
        return data.cpu()
    elif hasattr(data, 'to') and hasattr(data, 'device'):
        return data.to('cpu')
    elif isinstance(data, dict):
        return {k: recursive_to_cpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [recursive_to_cpu(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(recursive_to_cpu(v) for v in data)
    return data

try:
    from mmdet3d.models.fusion_models.bevfusion import BEVFusion
    MODELS.register_module(module=BEVFusion, name='BEVFusion', force=True)
    import mmdet3d.datasets.dataset_wrappers
    import mmdet3d.datasets.nuscenes_occupancy_dataset
    try:
        import mmdet3d.models.customs.cross_coordinate_sample
    except ImportError:
        pass
except ImportError as e:
    pass

def auto_fix_config(cfg, args=None):
    NOT_FOUND = object()
    def get_var_by_path(path, root_dict):
        try:
            if path in root_dict: return root_dict[path]
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
        if not isinstance(text, str) or '${' not in text: return text
        pattern = re.compile(r'\$\{(.+?)\}')
        while True:
            match = pattern.search(text)
            if not match: break
            full_str, content = match.group(0), match.group(1).strip()
            resolved = None
            val = get_var_by_path(content, cfg._cfg_dict)
            if val is not NOT_FOUND:
                if full_str == text: return val
                if isinstance(val, (str, int, float, bool)) or val is None: resolved = str(val)
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
                            valid = False;
                            break
                if valid: resolved = temp
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

class EmptyCacheHook(Hook):
    def after_train_epoch(self, runner):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class LegacyValidationHook(Hook):
    def __init__(self, val_dataloader_cfg, interval=1, show_dir=None):
        self.val_dataloader_cfg = val_dataloader_cfg
        self.interval = interval
        self.show_dir = show_dir
        self.dataloader = None
        self.dataset = None

    def after_train_epoch(self, runner):
        if self.interval > 0 and runner.epoch % self.interval == 0:
            self._run_validation(runner)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if get_dist_info()[0] == 0:
                    runner.logger.info(">>> [GC] CUDA Cache Cleared after Validation.")

    @torch.no_grad()
    def _run_validation(self, runner):
        runner.logger.info(f"\n>>> [AutoEval] Starting Evaluation for Epoch {runner.epoch}...")
        if self.dataloader is None:
            if 'dataset' in self.val_dataloader_cfg:
                self.val_dataloader_cfg['dataset']['test_mode'] = True
            ds_cfg = copy.deepcopy(self.val_dataloader_cfg['dataset'])
            self.dataset = DATASETS.build(ds_cfg)
            rank, world_size = get_dist_info()
            distributed = (world_size > 1)
            sampler = DistributedSampler(self.dataset, shuffle=False) if distributed else None
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=1,
                sampler=sampler,
                num_workers=2,
                collate_fn=naive_collate,
                pin_memory=True
            )

        model = runner.model
        model.eval()
        results = []
        rank, world_size = get_dist_info()

        if rank == 0:
            prog_bar = ProgressBar(len(self.dataloader))

        for i, data in enumerate(self.dataloader):

            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.cuda().float()  # 强制 FP32
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    data[k] = [x.cuda().float() for x in v]

            result = model(return_loss=False, rescale=True, **data)
            result = cast_tensor_to_float32(result)
            result_cpu = recursive_to_cpu(result)
            results.extend(result_cpu)
            if rank == 0: prog_bar.update()

        if world_size > 1:
            if rank == 0: runner.logger.info(f"\n>>> Gathering results from {world_size} GPUs...")
            collected_results = all_gather_object(results)
            final_results = []
            if rank == 0:
                max_len = max(len(r) for r in collected_results)
                for i in range(max_len):
                    for gpu_rank in range(world_size):
                        if i < len(collected_results[gpu_rank]):
                            final_results.append(collected_results[gpu_rank][i])
                if len(final_results) > len(self.dataset):
                    final_results = final_results[:len(self.dataset)]
        else:
            final_results = results
        if rank == 0:
            formatted_results = []
            has_boxes = False
            for res in final_results:
                new_res = recursive_to_cpu(res)

                if 'pts_bbox' not in new_res and 'boxes_3d' in new_res:
                    new_res['pts_bbox'] = {
                        'boxes_3d': new_res.get('boxes_3d'),
                        'scores_3d': new_res.get('scores_3d'),
                        'labels_3d': new_res.get('labels_3d')
                    }

                if 'pts_bbox' in new_res and new_res['pts_bbox']['boxes_3d'] is not None and len(
                        new_res['pts_bbox']['boxes_3d']) > 0:
                    has_boxes = True

                formatted_results.append(new_res)

            final_results = formatted_results

        if rank == 0:
            runner.logger.info(f"\n>>> Evaluating {len(final_results)} samples (Mode: FP32)...")
            try:
                if not has_boxes:
                    runner.logger.warning("\n" + "!" * 60)
                    runner.logger.warning(">>> WARNING: No valid bounding boxes detected in this epoch.")
                    runner.logger.warning(">>> Skipping NuScenes evaluation to prevent crash.")
                    runner.logger.warning("!" * 60 + "\n")
                else:
                    json_prefix = os.path.join(runner.work_dir, f'eval_epoch_{runner.epoch}')
                    eval_kwargs = dict(metric=['bbox'], jsonfile_prefix=json_prefix)
                    if self.show_dir: eval_kwargs['show_dir'] = self.show_dir

                    metrics = self.dataset.evaluate(final_results, **eval_kwargs)

                    runner.logger.info("\n" + "=" * 20 + " Evaluation Results " + "=" * 20)
                    if not metrics:
                        runner.logger.warning("!!! Metrics dictionary is empty.")
                    else:
                        for k, v in metrics.items():
                            k_lower = k.lower()
                            if any(x in k_lower for x in ['map', 'nds', 'miou', 'iou', 'recall', 'prec']):
                                if isinstance(v, float):
                                    runner.logger.info(f"{k}: {v:.4f}")
                                else:
                                    runner.logger.info(f"{k}: {v}")
                                if isinstance(v, (int, float)): runner.message_hub.update_scalar(f'val/{k}', v)
                    runner.logger.info("=" * 50 + "\n")

            except Exception as e:
                if "len(eval_boxes.boxes) > 0" in str(e):
                    runner.logger.warning(">>> [Suppressed Error] NuScenes Evaluation requires >0 boxes. Skipping...")
                else:
                    runner.logger.error(f"Evaluation Failed (Non-fatal): {e}")

        model.train()


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--auto-resume', action='store_true', help='resume from the latest checkpoint automatically')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='deterministic options')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args, opts = parser.parse_known_args()
    if 'LOCAL_RANK' not in os.environ: os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args, opts


def main():
    args, opts = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if args.launcher == 'none':
        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
            args.launcher = 'pytorch'
            if 'LOCAL_RANK' in os.environ:
                args.local_rank = int(os.environ['LOCAL_RANK'])

    if args.launcher != 'none':
        init_dist(args.launcher, backend='nccl')
        distributed = True
    else:
        distributed = False

    rank, _ = get_dist_info()
    if rank == 0:
        print(">>> [Info] Loading Config...")
        print(">>> [Info] Precision Mode: FP32 (TF32 DISABLED) for maximum accuracy.")
        if distributed:
            print(
                f">>> [Info] Distributed training ENABLED. Backend: nccl, World Size: {os.environ.get('WORLD_SIZE', '?')}")
        else:
            print(">>> [Warning] Distributed training is DISABLED. SyncBN will fall back to BN.")

    configs.load(args.config, recursive=True)
    if opts: configs.update(opts)
    cfg = Config(configs)
    cfg = auto_fix_config(cfg, args)

    cfg.launcher = args.launcher
    cfg.default_scope = 'mmdet3d'
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0], timestamp)

    if distributed:
        if cfg.get('sync_bn', 'torch') is not None:
            cfg.sync_bn = 'torch'
        if rank == 0:
            print(">>> [Info] SyncBN is enabled and forcing 'torch' implementation.")

    if torch.cuda.is_available() and not cfg.get('optim_wrapper'):
        dtype = 'float16'
        cfg.optim_wrapper = dict(
            type='AmpOptimWrapper', dtype=dtype, optimizer=cfg.optimizer,
            clip_grad=cfg.optimizer_config.get('grad_clip', None) if cfg.get('optimizer_config') else None
        )

    if cfg.get('data') and not cfg.get('train_dataloader'):
        if rank == 0: print(">>> [Info] Building train_dataloader...")
        cfg.train_dataloader = dict(
            batch_size=cfg.data.samples_per_gpu, num_workers=cfg.data.workers_per_gpu,
            persistent_workers=True, sampler=dict(type='DefaultSampler', shuffle=True),
            dataset=cfg.data.train, collate_fn=naive_collate
        )

    val_dataloader_cfg = None
    if cfg.get('val_dataloader'):
        val_dataloader_cfg = cfg.val_dataloader
    elif cfg.get('data') and cfg.data.get('val'):
        val_dataloader_cfg = dict(dataset=cfg.data.val, num_workers=cfg.data.workers_per_gpu)

    cfg.val_dataloader = None;
    cfg.val_cfg = None;
    cfg.val_evaluator = None

    if not cfg.get('default_hooks'):
        cfg.default_hooks = dict(
            timer=dict(type='IterTimerHook'), logger=dict(type='LoggerHook', interval=50),
            checkpoint=dict(type='CheckpointHook', interval=1), dist_sampler=dict(type='DistSamplerSeedHook'),
            param_scheduler=dict(type='ParamSchedulerHook')
        )

    cfg.custom_hooks = cfg.get('custom_hooks', [])
    cfg.custom_hooks.append(EmptyCacheHook())

    if val_dataloader_cfg is not None:
        eval_cfg = cfg.get('evaluation', {})
        eval_interval = eval_cfg.get('interval', 1) if isinstance(eval_cfg, dict) else 1
        hook = LegacyValidationHook(val_dataloader_cfg, interval=eval_interval)
        cfg.custom_hooks.append(hook)
        if rank == 0: print(f">>> [Info] Auto-Evaluation Enabled (Interval: {eval_interval} epochs)")

    if not cfg.get('train_cfg'):
        cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=cfg.runner.max_epochs, val_interval=999999)

    Config.pretty_text = property(lambda self: "Config print disabled.")

    resume_path = None

    if args.resume_from:
        resume_path = args.resume_from
    elif cfg.get('resume_from'):
        resume_path = cfg.get('resume_from')

    if resume_path:
        cfg.resume = True
        cfg.load_from = resume_path
        if rank == 0:
            print(f">>> [Info] Resuming training from: {resume_path}")
            print(f">>> [Info] Epoch/Optimizer state will be restored.")

    elif args.auto_resume:
        cfg.resume = True
        cfg.load_from = None
        if rank == 0:
            print(f">>> [Info]  Auto-resuming from latest checkpoint in work_dir...")

    if 'resume_from' in cfg:
        cfg.pop('resume_from')

    if rank == 0: print(">>> [Info] Building Runner...")
    if rank == 0:
        mm_logger = MMLogger.get_current_instance()
        original_level = mm_logger.level
        mm_logger.setLevel(logging.WARNING)

    runner = Runner.from_cfg(cfg)

    if rank == 0: mm_logger.setLevel(original_level)
    if args.seed is not None: runner.set_randomness(args.seed, deterministic=args.deterministic)
    if rank == 0:
        print("\n>>> [Patch] Injecting FP32 Safeguard for CenterHead (Forward + Loss)...")

    model_ref = runner.model
    if hasattr(model_ref, 'module'): model_ref = model_ref.module
    target_head = None
    if hasattr(model_ref, 'heads') and hasattr(model_ref.heads, 'object'):
        target_head = model_ref.heads.object
    elif hasattr(model_ref, 'bbox_head'):
        target_head = model_ref.bbox_head
    if target_head is not None:
        target_head.float()
        _orig_forward_single = target_head.forward_single

        def patched_forward_single(self, x):
            with torch.cuda.amp.autocast(enabled=False):
                x_fp32 = x.float()
                return _orig_forward_single(x_fp32)

        _orig_loss = target_head.loss

        def patched_loss(self, *args, **kwargs):
            with torch.cuda.amp.autocast(enabled=False):
                def to_f32(d):
                    if isinstance(d, torch.Tensor): return d.float()
                    if isinstance(d, list): return [to_f32(x) for x in d]
                    if isinstance(d, tuple): return tuple(to_f32(x) for x in d)
                    if isinstance(d, dict): return {k: to_f32(v) for k, v in d.items()}
                    return d

                args = to_f32(args)
                kwargs = to_f32(kwargs)
                return _orig_loss(*args, **kwargs)

        target_head.forward_single = types.MethodType(patched_forward_single, target_head)
        target_head.loss = types.MethodType(patched_loss, target_head)

        if rank == 0:
            print(">>> [Patch] Success! CenterHead is now completely isolated in FP32 domain.\n")
    else:
        if rank == 0:
            print(">>> [Warning] Could not find 'heads.object' or 'bbox_head'. Patch NOT applied.\n")
    runner.train()


if __name__ == '__main__':
    main()
