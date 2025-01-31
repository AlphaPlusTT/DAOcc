import torch


def get_reference_points(h_min, h_max, w_min, w_max, z_min, z_max, h_num, w_num, z_num,
                              device='cpu', dtype=torch.float):
    """Get the reference points used in BEVFormer.
    Args:
        h, w, z: spatial shape of volume.
        device (obj:`device`): The device where
            reference_points should be.
    Returns:
        Tensor: reference points used in decoder, has \
            shape (z_num * y_num * x_num, xyz).
    """
    z_offset = (z_max - z_min) / z_num * 0.5  # z
    w_offset = (w_max - w_min) / w_num * 0.5  # x
    h_offset = (h_max - h_min) / h_num * 0.5  # y
    zs = torch.linspace(z_min + z_offset, z_max - z_offset, z_num, dtype=dtype,
                        device=device).view(z_num, 1, 1).expand(z_num, h_num, w_num)  # z, y, x
    xs = torch.linspace(w_min + w_offset, w_max - w_offset, w_num, dtype=dtype,
                        device=device).view(1, 1, w_num).expand(z_num, h_num, w_num)  # z, y, x
    ys = torch.linspace(h_min + h_offset, h_max - h_offset, h_num, dtype=dtype,
                        device=device).view(1, h_num, 1).expand(z_num, h_num, w_num)  # z, y, x
    ref_3d = torch.stack((xs, ys, zs), -1)  # z_num, y_num, x_num, xyz
    ref_3d = ref_3d.permute(3, 0, 1, 2).flatten(1).permute(1, 0)
    return ref_3d  # z_num * y_num * x_num, xyz


