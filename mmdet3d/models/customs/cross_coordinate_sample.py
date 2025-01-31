from typing import List

import torch
import einops
import torch.nn.functional as F
from torch import nn
from mmcv.runner import BaseModule, force_fp32, auto_fp16
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer

from mmdet3d.models.builder import CUSTOMS
from ..utils.reference_poinst import get_reference_points

__all__ = ["CrossCoordinateSample"]


@CUSTOMS.register_module()
class CrossCoordinateSample(BaseModule):
    # TODO: [yz] make CrossCoordinateSample more generalized
    def __init__(self, point_range: list, point_num: list, lidar_point_range: list, point_type: str = 'ego',
                 extra_up: bool = False, extra_up_scale: int = 2, in_dim: int = 512, out_dim: int = 128,
                 norm_cfg=dict(type='BN')) -> None:
        """Convert points in ego/lidar coordinate to lidar coordinate and sample features corresponding to
        the converted points.
        Args:
            point_range (list[float]): [h_min, h_max, w_min, w_max, z_min, z_max], h<->y, w<->x.
            point_num (list[int]): [h_num, w_num, z_num], h<->y, w<->x.
            point_type (str): ego (Occ3D) or lidar (SurroundOcc and OpenOccupancy).
            lidar_point_range (list[float]): [y_min, y_max, x_min, x_max]
        """
        super().__init__()
        ref_points = get_reference_points(*point_range, *point_num)
        self.register_buffer('ref_points', ref_points)
        self.lidar_y_min, self.lidar_y_max, self.lidar_x_min, self.lidar_x_max = lidar_point_range
        self.output_h, self.output_w = point_num[:2]
        assert point_type in ['ego', 'lidar']
        self.point_type = point_type
        if in_dim != out_dim:
            self.transfer_conv = torch.nn.Conv2d(in_dim, out_dim, kernel_size=1)
        else:
            self.transfer_conv = None
        self.extra_up = extra_up
        if extra_up:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=extra_up_scale, mode='bilinear', align_corners=True),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_dim)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_dim, out_dim, kernel_size=1, padding=0)
            )
        self.fp16_enabled = False

    @force_fp32()
    def forward(self, x: torch.Tensor,
                lidar_aug_matrix: torch.Tensor, lidar2ego: torch.Tensor, occ_aug_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input features in lidar coordinate with shape (N, C, H, W).
            lidar_aug_matrix (torch.Tensor): (N, 4, 4)
            lidar2ego (torch.Tensor): (N, 4, 4)
            occ_aug_matrix (torch.Tensor): (N, 4, 4)
        Returns:
            x (torch.Tensor):
        """
        if self.transfer_conv is not None:
            x = self.transfer_conv(x)
        batch_size = x.shape[0]
        B = lidar_aug_matrix.shape[0]
        num_points = self.ref_points.shape[0]
        if self.point_type == 'ego':
            # inverse occ data augment
            # ([B 3 3] -> [B 1 3 3]) @ ([num_points 3] -> [1 num_points 3 1]) -> [B num_points 3 1]
            ref_ego = torch.inverse(occ_aug_matrix[:, :3, :3].view(B, 1, 3, 3)).matmul(self.ref_points.view(1, num_points, 3, 1))
            # [B num_points 3 1] -> [B num_points 3 1] -> [B num_points 3]
            ref_ego = ref_ego.view(B, num_points, 3, 1).squeeze(-1)

            # ego to lidar
            # [B num_points 3] - ([B 3] -> [B 1 3]) -> [B num_points 3] -> [B num_points 3 1]
            ref_lidar = (ref_ego - lidar2ego[:, :3, 3].view(batch_size, 1, 3)).unsqueeze(-1)
            # ([B 3 3] -> [B 3 3] -> [B 1 3 3]) @ [B num_points 3 1] -> [B num_points 3 1] -> [B num_points 3]
            # ref_lidar = lidar2ego[:, :3, :3].transpose(-1, -2).view(batch_size, 1, 3, 3).matmul(ref_lidar).squeeze(-1)
            # ([B 3 3] -> [B 1 3 3]) @ [B num_points 3 1] -> [B num_points 3 1]
            ref_lidar = torch.inverse(lidar2ego[:, :3, :3]).view(batch_size, 1, 3, 3).matmul(ref_lidar)

            # lidar data augment
            # ([B 3 3] -> [B 1 3 3]) @ [B num_points 3 1] -> [B num_points 3 1]
            ref_lidar = lidar_aug_matrix[:, :3, :3].view(B, 1, 3, 3).matmul(ref_lidar.view(B, num_points, 3, 1))
            # ([B num_points 3 1] -> [B num_points 3]) + ([B 3] -> [B 1 3]) -> [B num_points 3]
            ref_lidar = ref_lidar.squeeze(-1) + lidar_aug_matrix[:, :3, 3].unsqueeze(1)
            # [B num_points 3] -> [B num_points 1 3] -> [B num_points 1 2]
            ref_lidar = ref_lidar.unsqueeze(-2)[:, :, :, :2]
        else:  # self.point_type == 'lidar'
            # lidar data augment
            ref_lidar = self.ref_points[:, :2].unsqueeze(-2).unsqueeze(0).repeat(B, 1, 1, 1)

        lidar_x_length, lidar_y_length = self.lidar_x_max-self.lidar_x_min, self.lidar_y_max-self.lidar_y_min
        ref_lidar[..., 0] = (ref_lidar[..., 0] - self.lidar_x_min) / lidar_x_length
        ref_lidar[..., 1] = (ref_lidar[..., 1] - self.lidar_y_min) / lidar_y_length
        ref_lidar = ref_lidar * 2 - 1

        x = F.grid_sample(
            x,
            ref_lidar,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        x = einops.rearrange(x.squeeze(-1), 'bs c (h w) -> bs c h w', h=self.output_h, w=self.output_w)

        if self.extra_up:
            x = self.up(x)

        return x
