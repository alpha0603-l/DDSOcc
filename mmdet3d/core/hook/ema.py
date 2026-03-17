import math
import os
from copy import deepcopy
from glob import glob

import torch
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.dist import master_only
from mmengine.model import is_model_wrapper

def is_parallel(model):
    return is_model_wrapper(model)


class ModelEMA:

    def __init__(self, model, decay=0.9999, updates=0):
        self.ema_model = deepcopy(model).eval()
        self.ema = self.ema_model.module if is_parallel(self.ema_model) else self.ema_model
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, trainer, model):
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()

            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()


@HOOKS.register_module()
class MEGVIIEMAHook(Hook):

    def __init__(self, init_updates=0, decay=0.9990, resume=None, max_keep_ckpt=1):
        super().__init__()
        self.init_updates = init_updates
        self.resume = resume
        self.decay = decay
        self.max_keep_ckpt = max_keep_ckpt

    def before_run(self, runner):
        from torch.nn.modules.batchnorm import SyncBatchNorm

        bn_model_list = list()
        bn_model_dist_group_list = list()
        for model_ref in runner.model.modules():
            if isinstance(model_ref, SyncBatchNorm):
                bn_model_list.append(model_ref)
                bn_model_dist_group_list.append(model_ref.process_group)
                model_ref.process_group = None

        runner.ema_model = ModelEMA(runner.model, self.decay)

        for bn_model, dist_group in zip(bn_model_list,
                                        bn_model_dist_group_list):
            bn_model.process_group = dist_group
        runner.ema_model.updates = self.init_updates

        if self.resume is not None:
            runner.logger.info(f'resume ema checkpoint from {self.resume}')
            cpt = torch.load(self.resume, map_location='cpu')
            runner.ema_model.ema.load_state_dict(cpt['state_dict'], strict=False)
            runner.ema_model.updates = cpt['updates']

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        # TODO: [yz] start from 1/4 max epoch?
        current_model = runner.model.module if is_parallel(runner.model) else runner.model
        runner.ema_model.update(runner, current_model)

    def after_train_epoch(self, runner):
        self.save_checkpoint(runner)

    @master_only
    def save_checkpoint(self, runner):
        state_dict = runner.ema_model.ema.state_dict()
        ema_checkpoint = {
            'epoch': runner.epoch,
            'state_dict': state_dict,
            'updates': runner.ema_model.updates
        }
        save_path = f'epoch_{runner.epoch + 1}_ema.pth'
        save_path = os.path.join(runner.work_dir, save_path)
        torch.save(ema_checkpoint, save_path)
        runner.logger.info(f'Saving ema checkpoint at {save_path}')
        ckpt_files = glob(os.path.join(runner.work_dir, 'epoch_*_ema.pth'))
        try:
            ckpt_files.sort(key=lambda x: int(x.split('/')[-1].split('_')[1]))
        except:
            pass
        if len(ckpt_files) > self.max_keep_ckpt:
            for file in ckpt_files[:-self.max_keep_ckpt]:
                os.remove(file)
                runner.logger.info(f'Removing old checkpoint {file}')
