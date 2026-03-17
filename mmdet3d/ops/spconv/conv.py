import torch
from torch import nn
from mmengine.registry import MODELS

try:
    import spconv.pytorch as spconv
except ImportError:
    try:
        import spconv
    except ImportError:
        spconv = None

if spconv is None:
    print("⚠️ Warning: spconv is NOT installed! Sparse layers will fail at runtime.")

def register_spconv_module(name):
    def decorator(cls):
        MODELS.register_module(name=name, module=cls, force=True)
        return cls

    return decorator


if spconv is not None:
    @register_spconv_module("SparseConv2d")
    class SparseConv2d(spconv.SparseConv2d):
        def forward(self, input): return super().forward(input)

    @register_spconv_module("SparseConv3d")
    class SparseConv3d(spconv.SparseConv3d):
        def forward(self, input): return super().forward(input)

    @register_spconv_module("SubMConv2d")
    class SubMConv2d(spconv.SubMConv2d):
        def forward(self, input): return super().forward(input)

    @register_spconv_module("SubMConv3d")
    class SubMConv3d(spconv.SubMConv3d):
        def forward(self, input): return super().forward(input)

    @register_spconv_module("SparseConvTranspose2d")
    class SparseConvTranspose2d(spconv.SparseConvTranspose2d):
        def forward(self, input): return super().forward(input)


    @register_spconv_module("SparseConvTranspose3d")
    class SparseConvTranspose3d(spconv.SparseConvTranspose3d):
        def forward(self, input): return super().forward(input)

    @register_spconv_module("SparseInverseConv2d")
    class SparseInverseConv2d(spconv.SparseInverseConv2d):
        def forward(self, input): return super().forward(input)

    @register_spconv_module("SparseInverseConv3d")
    class SparseInverseConv3d(spconv.SparseInverseConv3d):
        def forward(self, input): return super().forward(input)

    @register_spconv_module("SparseMaxPool2d")
    class SparseMaxPool2d(spconv.SparseMaxPool2d):
        def forward(self, input): return super().forward(input)

    @register_spconv_module("SparseMaxPool3d")
    class SparseMaxPool3d(spconv.SparseMaxPool3d):
        def forward(self, input): return super().forward(input)

    class SparseSequential(spconv.SparseSequential):
        def forward(self, input): return super().forward(input)

    class SparseModule(spconv.SparseModule):
        def forward(self, input): return super().forward(input)

else:
    class MockModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise NotImplementedError("spconv library is not installed!")


    SparseConv2d = MODELS.register_module("SparseConv2d", module=MockModule, force=True)
    SparseConv3d = MODELS.register_module("SparseConv3d", module=MockModule, force=True)
    SubMConv2d = MODELS.register_module("SubMConv2d", module=MockModule, force=True)
    SubMConv3d = MODELS.register_module("SubMConv3d", module=MockModule, force=True)
    SparseConvTranspose2d = MODELS.register_module("SparseConvTranspose2d", module=MockModule, force=True)
    SparseConvTranspose3d = MODELS.register_module("SparseConvTranspose3d", module=MockModule, force=True)
    SparseInverseConv2d = MODELS.register_module("SparseInverseConv2d", module=MockModule, force=True)
    SparseInverseConv3d = MODELS.register_module("SparseInverseConv3d", module=MockModule, force=True)
    SparseMaxPool2d = MODELS.register_module("SparseMaxPool2d", module=MockModule, force=True)
    SparseMaxPool3d = MODELS.register_module("SparseMaxPool3d", module=MockModule, force=True)
    SparseSequential = SparseModule = MockModule
