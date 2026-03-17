from mmengine.model import is_model_wrapper as is_parallel
from .ema import MEGVIIEMAHook
try:
    from .sequentialcontrol import SequentialControlHook
except ImportError:
    SequentialControlHook = None

try:
    from .syncbncontrol import SyncbnControlHook
except ImportError:
    SyncbnControlHook = None

__all__ = ['MEGVIIEMAHook', 'SequentialControlHook', 'is_parallel',
           'SyncbnControlHook']
