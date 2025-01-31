# Copyright (c) OpenMMLab. All rights reserved.
import einops
import torch
from torch import nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
import numpy as np
from mmdet3d.models.builder import HEADS, build_loss, build_custom

__all__ = ["BEVOCCHead2D"]

nusc_class_frequencies = np.array([
    944004,
    1897170,
    152386,
    2391677,
    16957802,
    724139,
    189027,
    2074468,
    413451,
    2384460,
    5916653,
    175883646,
    4275424,
    51393615,
    61411620,
    105975596,
    116424404,
    1892500630
])


@HEADS.register_module()
class BEVOCCHead2D(BaseModule):
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 Dz=16,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True,
                 class_balance=False,
                 loss_occ=None,
                 coordinate_transform=None
                 ):
        super(BEVOCCHead2D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        out_channels = out_dim if use_predicter else num_classes * Dz
        self.final_conv = ConvModule(
            self.in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes * Dz),
            )

        self.use_mask = use_mask
        self.num_classes = num_classes

        self.class_balance = class_balance
        if self.class_balance:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
            self.cls_weights = class_weights
            loss_occ['class_weight'] = class_weights        # ce loss
        self.loss_occ = build_loss(loss_occ)
        if coordinate_transform is not None:
            self.coordinate_transform = build_custom(coordinate_transform)
        else:
            self.coordinate_transform = None

    def forward(self, img_feats, lidar_aug_matrix=None, lidar2ego=None, occ_aug_matrix=None):
        """
        Args:
            img_feats: (B, C, Dx, Dy)

        Returns:

        """
        if isinstance(img_feats, list):
            assert len(img_feats) == 1
            img_feats = img_feats[0]
        img_feats = einops.rearrange(img_feats, 'bs c w h -> bs c h w')
        if self.coordinate_transform is not None:
            img_feats = self.coordinate_transform(img_feats, lidar_aug_matrix, lidar2ego, occ_aug_matrix)
        # (B, C, Dy, Dx) --> (B, C, Dy, Dx) --> (B, Dx, Dy, C)
        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
        if self.use_predicter:
            # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*n_cls)
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)

        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        loss = dict()
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)   # (B, Dx, Dy, Dz)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            voxel_semantics = voxel_semantics.reshape(-1)
            # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, n_cls)
            preds = occ_pred.reshape(-1, self.num_classes)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            mask_camera = mask_camera.reshape(-1)

            if self.class_balance:
                valid_voxels = voxel_semantics[mask_camera.bool()]
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (valid_voxels == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = mask_camera.sum()

            loss_occ = self.loss_occ(
                preds,      # (B*Dx*Dy*Dz, n_cls)
                voxel_semantics,    # (B*Dx*Dy*Dz, )
                mask_camera,        # (B*Dx*Dy*Dz, )
                avg_factor=num_total_samples
            )
            loss['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = occ_pred.reshape(-1, self.num_classes)

            if self.class_balance:
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (voxel_semantics == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = len(voxel_semantics)

            loss_occ = self.loss_occ(
                preds,
                voxel_semantics,
                avg_factor=num_total_samples,
            )

            loss['loss_occ'] = loss_occ
        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)
