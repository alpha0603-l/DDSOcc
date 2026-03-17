import torch
from mmdet.registry import TASK_UTILS
from mmengine.utils import is_list_of

@TASK_UTILS.register_module()
class Anchor3DRangeGenerator:
    """3D Anchor Generator by range.

    This anchor generator generates anchors by the given range in different
    feature levels.
    Due the convention in 3D detection, different anchor sizes are related to
    different ranges for different categories. However we find this setting
    does not effect the performance much in some datasets, e.g., nuScenes.

    Args:
        ranges (list[list[float]]): Ranges of different anchors.
            The ranges are the same across different feature levels. But may
            vary for different anchor sizes if size_per_range is True.
        sizes (list[list[float]]): 3D sizes of anchors.
        scales (list[int]): Scales of anchors in different feature levels.
        rotations (list[float]): Rotations of anchors in a feature grid.
        custom_values (tuple[float]): Customized values of that anchor. For
            example, in nuScenes the anchors have velocities.
        reshape_out (bool): Whether to reshape the output into (N x 4).
        size_per_range: Whether to use separate ranges for different sizes.
            If size_per_range is True, the ranges should have the same length
            as the sizes, if not, it will be duplicated.
    """

    def __init__(
        self,
        ranges,
        sizes=[[1.6, 3.9, 1.56]],
        scales=[1],
        rotations=[0, 1.5707963],
        custom_values=(),
        reshape_out=True,
        size_per_range=True,
    ):
        assert is_list_of(ranges, list)
        if size_per_range:
            if len(sizes) != len(ranges):
                assert len(ranges) == 1
                ranges = ranges * len(sizes)
            assert len(ranges) == len(sizes)
        else:
            assert len(ranges) == 1
        assert is_list_of(sizes, list)
        assert isinstance(scales, list)

        self.sizes = sizes
        self.scales = scales
        self.ranges = ranges
        self.rotations = rotations
        self.custom_values = custom_values
        self.cached_anchors = None
        self.reshape_out = reshape_out
        self.size_per_range = size_per_range

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += f"anchor_range={self.ranges},\n"
        s += f"scales={self.scales},\n"
        s += f"sizes={self.sizes},\n"
        s += f"rotations={self.rotations},\n"
        s += f"reshape_out={self.reshape_out},\n"
        s += f"size_per_range={self.size_per_range})"
        return s

    @property
    def num_base_anchors(self):
        """list[int]: Total number of base anchors in a feature grid."""
        num_rot = len(self.rotations)
        num_size = torch.tensor(self.sizes).reshape(-1, 3).size(0)
        return num_rot * num_size

    @property
    def num_levels(self):
        """int: Number of feature levels that the generator is applied to."""
        return len(self.scales)

    def grid_anchors(self, featmap_sizes, device="cuda"):
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(
                featmap_sizes[i], self.scales[i], device=device
            )
            if self.reshape_out:
                anchors = anchors.reshape(-1, anchors.size(-1))
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_anchors(self, featmap_size, scale, device="cuda"):
        if not self.size_per_range:
            return self.anchors_single_range(
                featmap_size,
                self.ranges[0],
                scale,
                self.sizes,
                self.rotations,
                device=device,
            )

        mr_anchors = []
        for anchor_range, anchor_size in zip(self.ranges, self.sizes):
            mr_anchors.append(
                self.anchors_single_range(
                    featmap_size,
                    anchor_range,
                    scale,
                    anchor_size,
                    self.rotations,
                    device=device,
                )
            )
        mr_anchors = torch.cat(mr_anchors, dim=-3)
        return mr_anchors

    def anchors_single_range(
        self,
        feature_size,
        anchor_range,
        scale=1,
        sizes=[[1.6, 3.9, 1.56]],
        rotations=[0, 1.5707963],
        device="cuda",
    ):
        if len(feature_size) == 2:
            feature_size = [1, feature_size[0], feature_size[1]]
        anchor_range = torch.tensor(anchor_range, device=device)
        z_centers = torch.linspace(
            anchor_range[2], anchor_range[5], feature_size[0], device=device
        )
        y_centers = torch.linspace(
            anchor_range[1], anchor_range[4], feature_size[1], device=device
        )
        x_centers = torch.linspace(
            anchor_range[0], anchor_range[3], feature_size[2], device=device
        )
        sizes = torch.tensor(sizes, device=device).reshape(-1, 3) * scale
        rotations = torch.tensor(rotations, device=device)
        rets = torch.meshgrid(x_centers, y_centers, z_centers, rotations)
        rets = list(rets)
        tile_shape = [1] * 5
        tile_shape[-2] = int(sizes.shape[0])
        for i in range(len(rets)):
            rets[i] = rets[i].unsqueeze(-2).repeat(tile_shape).unsqueeze(-1)

        sizes = sizes.reshape([1, 1, 1, -1, 1, 3])
        tile_size_shape = list(rets[0].shape)
        tile_size_shape[3] = 1
        sizes = sizes.repeat(tile_size_shape)
        rets.insert(3, sizes)

        ret = torch.cat(rets, dim=-1).permute([2, 1, 0, 3, 4, 5])
        if len(self.custom_values) > 0:
            custom_ndim = len(self.custom_values)
            custom = ret.new_zeros([*ret.shape[:-1], custom_ndim])
            ret = torch.cat([ret, custom], dim=-1)
        return ret


@TASK_UTILS.register_module()
class AlignedAnchor3DRangeGenerator(Anchor3DRangeGenerator):

    def __init__(self, align_corner=False, **kwargs):
        super(AlignedAnchor3DRangeGenerator, self).__init__(**kwargs)
        self.align_corner = align_corner

    def anchors_single_range(
        self,
        feature_size,
        anchor_range,
        scale,
        sizes=[[1.6, 3.9, 1.56]],
        rotations=[0, 1.5707963],
        device="cuda",
    ):
        if len(feature_size) == 2:
            feature_size = [1, feature_size[0], feature_size[1]]
        anchor_range = torch.tensor(anchor_range, device=device)
        z_centers = torch.linspace(
            anchor_range[2], anchor_range[5], feature_size[0] + 1, device=device
        )
        y_centers = torch.linspace(
            anchor_range[1], anchor_range[4], feature_size[1] + 1, device=device
        )
        x_centers = torch.linspace(
            anchor_range[0], anchor_range[3], feature_size[2] + 1, device=device
        )
        sizes = torch.tensor(sizes, device=device).reshape(-1, 3) * scale
        rotations = torch.tensor(rotations, device=device)

        if not self.align_corner:
            z_shift = (z_centers[1] - z_centers[0]) / 2
            y_shift = (y_centers[1] - y_centers[0]) / 2
            x_shift = (x_centers[1] - x_centers[0]) / 2
            z_centers += z_shift
            y_centers += y_shift
            x_centers += x_shift

        rets = torch.meshgrid(
            x_centers[: feature_size[2]],
            y_centers[: feature_size[1]],
            z_centers[: feature_size[0]],
            rotations,
        )

        rets = list(rets)
        tile_shape = [1] * 5
        tile_shape[-2] = int(sizes.shape[0])
        for i in range(len(rets)):
            rets[i] = rets[i].unsqueeze(-2).repeat(tile_shape).unsqueeze(-1)

        sizes = sizes.reshape([1, 1, 1, -1, 1, 3])
        tile_size_shape = list(rets[0].shape)
        tile_size_shape[3] = 1
        sizes = sizes.repeat(tile_size_shape)
        rets.insert(3, sizes)

        ret = torch.cat(rets, dim=-1).permute([2, 1, 0, 3, 4, 5])

        if len(self.custom_values) > 0:
            custom_ndim = len(self.custom_values)
            custom = ret.new_zeros([*ret.shape[:-1], custom_ndim])
            # TODO: check the support of custom values
            # custom[:] = self.custom_values
            ret = torch.cat([ret, custom], dim=-1)
        return ret


@TASK_UTILS.register_module()
class AlignedAnchor3DRangeGeneratorPerCls(AlignedAnchor3DRangeGenerator):
    def __init__(self, **kwargs):
        super(AlignedAnchor3DRangeGeneratorPerCls, self).__init__(**kwargs)
        assert len(self.scales) == 1, (
            "Multi-scale feature map levels are"
            + " not supported currently in this kind of anchor generator."
        )

    def grid_anchors(self, featmap_sizes, device="cuda"):
        multi_level_anchors = []
        anchors = self.multi_cls_grid_anchors(
            featmap_sizes, self.scales[0], device=device
        )
        multi_level_anchors.append(anchors)
        return multi_level_anchors

    def multi_cls_grid_anchors(self, featmap_sizes, scale, device="cuda"):
        assert len(featmap_sizes) == len(self.sizes) == len(self.ranges), (
            "The number of different feature map sizes anchor sizes and "
            + "ranges should be the same."
        )

        multi_cls_anchors = []
        for i in range(len(featmap_sizes)):
            anchors = self.anchors_single_range(
                featmap_sizes[i],
                self.ranges[i],
                scale,
                self.sizes[i],
                self.rotations,
                device=device,
            )
            ndim = len(featmap_sizes[i])
            anchors = anchors.view(*featmap_sizes[i], -1, anchors.size(-1))
            anchors = anchors.permute(ndim, *range(0, ndim), ndim + 1)
            multi_cls_anchors.append(anchors.reshape(-1, anchors.size(-1)))
        return multi_cls_anchors
