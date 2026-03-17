from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

@HOOKS.register_module()
class SequentialControlHook(Hook):

    def __init__(self, temporal_start_epoch=1):
        super().__init__()
        self.temporal_start_epoch = temporal_start_epoch

    def set_temporal_flag(self, runner, flag):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        if is_model_wrapper(model):
            model = model.module
        try:
            model.with_prev = flag
        except AttributeError:
            pass

    def before_run(self, runner):
        self.set_temporal_flag(runner, False)

    def before_train_epoch(self, runner):
        if runner.epoch > self.temporal_start_epoch:
            self.set_temporal_flag(runner, True)
