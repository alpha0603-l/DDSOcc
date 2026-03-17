from torch import nn

__all__ = ['is_parallel']

def is_parallel(model):
    parallel_type = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)
