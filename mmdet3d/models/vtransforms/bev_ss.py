from typing import Tuple, List

import torch
from mmcv.runner import BaseModule, force_fp32, auto_fp16
from torch import nn
import torch.nn.functional as F
import einops
from mmdet3d.models.builder import VTRANSFORMS
from ..utils.reference_poinst import get_reference_points

__all__ = ["BEVTransform"]


@VTRANSFORMS.register_module()
class BEVTransform(BaseModule):
    r"""
    Please refer to the paper DAOcc['https://arxiv.org/abs/2409.19972'].

    Args:
        x, y, z: Range in three directions.
        xs, ys, zs: Resolution in three directions.
        input_size (tuple(int)): Size of input images in format of (height, width).
        in_channels (int): Channels of input feature.
        out_channels (int): Channels of transformed feature.
        top_type (str): Coordinate system in which the task is labeled, 'lidar' or 'ego',
                        eg: 'lidar' for SurroundOcc and 'ego' for Occ3D
    """

    def __init__(
            self,
            x: List[float], y: List[float], z: List[float],
            xs: int, ys: int, zs: int,
            input_size: List[int],
            in_channels: int = 256,
            out_channels: int = 128,
            top_type: str = 'lidar',
            down_sample: bool = False,
            down_sample_scale: int = 2,
            down_sample_channels: List[int] = [128*10, 64*10, 32*10, 16*10]
    ):
        super(BEVTransform, self).__init__()
        self.pc_range = [x[0], y[0], z[0], x[1], y[1], z[1]]
        self.volume_size = [int(s) for s in [xs, ys, zs]]
        ref_3d = get_reference_points(self.pc_range[0], self.pc_range[3],
                                      self.pc_range[1], self.pc_range[4],
                                      self.pc_range[2], self.pc_range[5],
                                      self.volume_size[0], self.volume_size[1], self.volume_size[2])
        self.register_buffer('ref_3d', ref_3d)
        self.transfer_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.initial_flag = True
        self.input_size = input_size
        self.top_type = top_type
        if down_sample:
            assert down_sample_scale in [1, 2], 'only support 1x or 2x down sampling!'
            if down_sample_scale == 2:
                self.down_sample = nn.Sequential(
                    nn.Conv2d(down_sample_channels[0], down_sample_channels[1], 3, padding=1, bias=False),
                    nn.BatchNorm2d(down_sample_channels[1]),
                    nn.ReLU(True),
                    nn.Conv2d(
                        down_sample_channels[1],
                        down_sample_channels[2],
                        3,
                        stride=down_sample_scale,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(down_sample_channels[2]),
                    nn.ReLU(True),
                    nn.Conv2d(down_sample_channels[2], down_sample_channels[3], 3, padding=1, bias=False),
                    nn.BatchNorm2d(down_sample_channels[3]),
                    nn.ReLU(True),
                )
            else:  # down_sample_scale == 1
                assert len(down_sample_channels) == 2, "the len of 'down_sample_channels: List[int]' should be 2!"
                self.down_sample = nn.Sequential(
                    nn.Conv2d(down_sample_channels[0], down_sample_channels[1], 3, padding=1, bias=False),
                    nn.BatchNorm2d(down_sample_channels[1]),
                    nn.ReLU(True)
                )
        else:
            self.down_sample = None
        self.fp16_enabled = False

    @force_fp32()
    def forward(self,
                x,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                **kwargs
                ):
        """Transform image-view feature into bird-eye-view feature.

        Args:
        Returns:
        """
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.transfer_conv(x)
        _, C, _, _ = x.shape

        if self.top_type == 'ego':
            assert 'camera_ego2global' in kwargs
            camera_ego2global = kwargs['camera_ego2global']
            keyego2global = camera_ego2global[:, 0, ...].unsqueeze(1)  # (B, 1, 4, 4)
            global2keyego = torch.inverse(keyego2global.double())  # (B, 1, 4, 4)
            camera2sensor = global2keyego @ camera_ego2global.double() @ camera2ego.double()  # (B, N_views, 4, 4)
            camera2sensor = camera2sensor.float()
        elif self.top_type == 'lidar':
            camera2sensor = camera2lidar
        else:
            raise NotImplementedError
        reference_points_img, volume_mask = self.point_sampling(camera2sensor, camera_intrinsics[..., :3, :3],
                                                                img_aug_matrix[..., :3, :3], img_aug_matrix[..., :3, 3],
                                                                lidar_aug_matrix)
        num_cams, bs, num_voxels, num_points_in_voxel, _ = reference_points_img.shape
        indexes = [[] for _ in range(bs)]
        for i, mask_per_img in enumerate(volume_mask):
            for j in range(bs):
                assert mask_per_img[j].shape[-1] == 1
                index_query_per_img = mask_per_img[j].squeeze(-1).nonzero().squeeze(-1)
                indexes[j].append(index_query_per_img)
        all_indexes = [item for sublist in indexes for item in sublist]
        max_len = max([len(each) for each in all_indexes])

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        reference_points_rebatch = reference_points_img.new_zeros([bs, num_cams, max_len, num_points_in_voxel, 2])
        for j in range(bs):
            indexes_per_batch = indexes[j]
            for ii, reference_points_per_img in enumerate(reference_points_img):
                index_query_per_img = indexes_per_batch[ii]
                reference_points_rebatch[j, ii, :len(index_query_per_img)] = reference_points_per_img[
                    j, index_query_per_img]
        reference_points_rebatch = reference_points_rebatch.view(bs * num_cams, max_len, num_points_in_voxel, 2)
        reference_points_rebatch = 2 * reference_points_rebatch - 1

        # x: bn, c, h, w
        # reference_points_rebatch: bn, max_len, num_points_in_voxel, uv/xy/wh
        sampling_feats = F.grid_sample(
            x,
            reference_points_rebatch,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        # bn, c, max_len, num_points_in_voxel
        sampling_feats = einops.rearrange(sampling_feats,
                                          '(b n) c max_len d -> b n max_len (d c)',
                                          b=bs)
        feats_volume = x.new_zeros([bs, num_voxels, C])
        for j in range(bs):
            indexes_per_batch = indexes[j]
            for ii, index_query_per_img in enumerate(indexes_per_batch):
                feats_volume[j, index_query_per_img] += sampling_feats[j, ii, :len(index_query_per_img)]
        count = volume_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        feats_volume = feats_volume / count[..., None]
        feats_volume = einops.rearrange(feats_volume,
                                        'bs (z h w) c -> bs (z c) w h',
                                        z=self.volume_size[2], h=self.volume_size[0], w=self.volume_size[1])
        if self.down_sample is not None:
            feats_volume = self.down_sample(feats_volume)
        return feats_volume

    @force_fp32()
    def point_sampling(self, camera2sensor, cam2imgs, post_rots, post_trans, bda):
        B, N, _, _ = camera2sensor.shape
        num_points = self.ref_3d.shape[0]

        # [(NP 3) -> (1 NP 3)] - [(B 3) -> (B 1 3)] -> (B NP 3)
        reference_points = self.ref_3d.view(1, num_points, 3) - bda[:, :3, 3].view(B, 1, 3)
        # [(B 3 3) -> (B 1 3 3)] @ [(B NP 3) -> (B NP 3 1)] -> (B NP 3 1)
        reference_points = torch.inverse(bda[:, :3, :3].view(B, 1, 3, 3)).matmul(reference_points.unsqueeze(-1))
        # (B NP 3 1) -> (B 1 NP 3 1) -> (B 1 NP 3)
        reference_points = reference_points.view(B, 1, num_points, 3, 1).squeeze(-1)
        # (B 1 NP 3) - (B N 1 3) -> (B N NP 3) -> (B N NP 3 1)
        reference_points = (reference_points - camera2sensor[:, :, :3, 3].view(B, N, 1, 3)).unsqueeze(-1)
        # (B N 3 3)
        combine = cam2imgs.matmul(camera2sensor[:, :, :3, :3].transpose(-1, -2))
        # [(B N 3 3) -> (B N 1 3 3)] @ (B N NP 3 1) -> (B N NP 3 1) -> (B N NP 3)
        reference_points_img = combine.view(B, N, 1, 3, 3).matmul(reference_points).squeeze(-1)

        eps = 1e-5
        # (B N NP 1)
        volume_mask = (reference_points_img[..., 2:3] > eps)
        # (B N NP 2)
        reference_points_img = reference_points_img[..., 0:2] / torch.maximum(
            reference_points_img[..., 2:3], torch.ones_like(reference_points_img[..., 2:3]) * eps)

        # do post-transformation
        post_rots2 = post_rots[:, :, :2, :2]
        post_trans2 = post_trans[:, :, :2]
        # [(B N 2 2) -> (B N 1 2 2)] @ [(B N NP 2) -> (B N NP 2 1)] -> (B N NP 2 1)
        reference_points_img = post_rots2.view(B, N, 1, 2, 2).matmul(reference_points_img.unsqueeze(-1))
        # [(B N NP 2 1) -> (B N NP 2)] + [(B N 2) -> (B N 1 2)] -> (B N NP 2)
        reference_points_img = reference_points_img.squeeze(-1) + post_trans2.view(B, N, 1, 2)

        H_in, W_in = self.input_size
        reference_points_img[..., 0] /= W_in
        reference_points_img[..., 1] /= H_in

        volume_mask = (volume_mask & (reference_points_img[..., 1:2] > 0.0)
                       & (reference_points_img[..., 1:2] < 1.0)
                       & (reference_points_img[..., 0:1] < 1.0)
                       & (reference_points_img[..., 0:1] > 0.0))

        volume_mask = torch.nan_to_num(volume_mask)

        reference_points_img = reference_points_img.unsqueeze(-2).permute(1, 0, 2, 3, 4)  # num_cam, B, num_query, D, uv
        volume_mask = volume_mask.permute(1, 0, 2, 3)

        return reference_points_img, volume_mask
