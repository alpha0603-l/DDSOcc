import torch
import torch.nn as nn
from torch import distributed as dist
from torch.autograd.function import Function
from mmengine.registry import MODELS
def force_fp32(apply_to=None, out_fp16=False):
    def decorator(func):
        def wrapper(self, input, *args, **kwargs):
            if input.dtype != torch.float32:
                input = input.float()
            ret = func(self, input, *args, **kwargs)
            if out_fp16 and torch.cuda.is_available() and input.is_cuda:
                pass
            return ret
        return wrapper
    return decorator

class AllReduce(Function):
    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)
    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output

@MODELS.register_module("naiveSyncBN1d")
class NaiveSyncBatchNorm1d(nn.BatchNorm1d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp16_enabled = False
    @force_fp32(out_fp16=True)
    def forward(self, input):
        assert (
                input.dtype == torch.float32
        ), f"input should be in float32 type, got {input.dtype}"
        if not dist.is_initialized() or dist.get_world_size() == 1 or not self.training:
            return super().forward(input)
        assert input.shape[0] > 0, "SyncBN does not support empty inputs"
        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2])
        meansqr = torch.mean(input * input, dim=[0, 2])
        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())
        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)
        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1)
        bias = bias.reshape(1, -1, 1)
        return input * scale + bias


@MODELS.register_module("naiveSyncBN2d")
class NaiveSyncBatchNorm2d(nn.BatchNorm2d):
    """Syncronized Batch Normalization for 4D Tensors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp16_enabled = False
    @force_fp32(out_fp16=True)
    def forward(self, input):
        assert (
                input.dtype == torch.float32
        ), f"input should be in float32 type, got {input.dtype}"

        if not dist.is_initialized() or dist.get_world_size() == 1 or not self.training:
            return super().forward(input)
        assert input.shape[0] > 0, "SyncBN does not support empty inputs"
        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return input * scale + bias
