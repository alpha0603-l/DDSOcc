from torch.nn import SyncBatchNorm
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

__all__ = ['SyncbnControlHook']


@HOOKS.register_module()
class SyncbnControlHook(Hook):

    def __init__(self, syncbn_start_epoch=1):
        super().__init__()
        self.is_syncbn = False
        self.syncbn_start_epoch = syncbn_start_epoch

    def cvt_syncbn(self, runner):
        model = runner.model
        if is_model_wrapper(model):
            converted_module = SyncBatchNorm.convert_sync_batchnorm(
                model.module,
                process_group=None
            )
            model.module = converted_module
        else:
            converted_module = SyncBatchNorm.convert_sync_batchnorm(
                model,
                process_group=None
            )
            runner.model = converted_module

    def before_train_epoch(self, runner):
        if runner.epoch >= self.syncbn_start_epoch and not self.is_syncbn:
            runner.logger.info(f'SyncbnControlHook: Switching to SyncBatchNorm at epoch {runner.epoch}')
            self.cvt_syncbn(runner)
            self.is_syncbn = True
