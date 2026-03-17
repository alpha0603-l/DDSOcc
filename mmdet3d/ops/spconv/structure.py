import torch

try:
    import spconv.pytorch as spconv
except ImportError:
    try:
        import spconv
    except ImportError:
        spconv = None

if spconv is not None:
    SparseConvTensor = spconv.SparseConvTensor
else:
    class SparseConvTensor:
        def __init__(self, features, indices, spatial_shape, batch_size):
            self.features = features
            self.indices = indices
            self.spatial_shape = spatial_shape
            self.batch_size = batch_size

def scatter_point(features, indices, spatial_shape):
    pass
