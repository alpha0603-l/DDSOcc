import os
import numpy as np
import torch

__all__ = ["load_augmented_point_cloud", "reduce_LiDAR_beams"]


def load_augmented_point_cloud(path, virtual=False, reduce_beams=32):
    """Load augmented point cloud with virtual points."""
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    tokens = path.split("/")
    vp_dir = "_VIRTUAL" if reduce_beams == 32 else f"_VIRTUAL_{reduce_beams}BEAMS"
    try:
        if path.startswith('/'):
            base_tokens = ['/'] + tokens[:-3]
        else:
            base_tokens = tokens[:-3]

        seg_path = os.path.join(
            *base_tokens,
            "virtual_points",
            tokens[-3],
            tokens[-2] + vp_dir,
            tokens[-1] + ".pkl.npy",
        )
    except IndexError:
        raise ValueError(f"Path structure unexpected: {path}")
    if reduce_beams is not None:
        if not os.path.exists(seg_path):
            pass

    assert os.path.exists(seg_path), f"Virtual point file not found: {seg_path}"
    data_dict = np.load(seg_path, allow_pickle=True).item()

    virtual_points1 = data_dict["real_points"]
    virtual_points2 = np.concatenate(
        [
            data_dict["virtual_points"][:, :3],
            np.zeros([data_dict["virtual_points"].shape[0], 1]),
            data_dict["virtual_points"][:, 3:],
        ],
        axis=-1,
    )

    points = np.concatenate(
        [
            points,
            np.ones([points.shape[0], virtual_points1.shape[1] - points.shape[1] + 1]),
        ],
        axis=1,
    )
    virtual_points1 = np.concatenate(
        [virtual_points1, np.zeros([virtual_points1.shape[0], 1])], axis=1
    )

    if len(data_dict["real_points_indice"]) > 0:
        points[data_dict["real_points_indice"]] = virtual_points1

    if virtual:
        virtual_points2 = np.concatenate(
            [virtual_points2, -1 * np.ones([virtual_points2.shape[0], 1])], axis=1
        )
        points = np.concatenate([points, virtual_points2], axis=0).astype(np.float32)
    return points


def reduce_LiDAR_beams(pts, reduce_beams_to=32):
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts)

    radius = torch.sqrt(pts[:, 0].pow(2) + pts[:, 1].pow(2) + pts[:, 2].pow(2))
    sine_theta = pts[:, 2] / radius
    theta = torch.asin(sine_theta)

    top_ang = 0.1862
    down_ang = -0.5353

    beam_range = torch.zeros(32)
    beam_range[0] = top_ang
    beam_range[31] = down_ang

    for i in range(1, 31):
        beam_range[i] = beam_range[i - 1] - 0.023275

    num_pts, _ = pts.size()
    mask = torch.zeros(num_pts)

    if reduce_beams_to == 16:
        for id in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]:
            beam_mask = (theta < (beam_range[id - 1] - 0.012)) * (
                    theta > (beam_range[id] - 0.012)
            )
            mask = mask + beam_mask
        mask = mask.bool()
    elif reduce_beams_to == 4:
        for id in [7, 9, 11, 13]:
            beam_mask = (theta < (beam_range[id - 1] - 0.012)) * (
                    theta > (beam_range[id] - 0.012)
            )
            mask = mask + beam_mask
        mask = mask.bool()
    elif reduce_beams_to == 1:
        chosen_beam_id = 9
        mask = (theta < (beam_range[chosen_beam_id - 1] - 0.012)) * (
                theta > (beam_range[chosen_beam_id] - 0.012)
        )
        mask = mask.bool()
    else:
        raise NotImplementedError(f"reduce_beams_to={reduce_beams_to} not supported")

    points = pts[mask]
    return points.numpy()
