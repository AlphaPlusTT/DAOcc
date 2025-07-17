import tempfile
import os
from typing import Any, Dict
import cv2
import mmcv
import numpy as np
import pyquaternion
import torch
from nuscenes.utils.data_classes import Box as NuScenesBox
from pyquaternion import Quaternion
from tqdm import tqdm

from mmdet.datasets import DATASETS

from ..core.bbox import LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset
from .nuscenes_dataset import NuScenesDataset
from ..core.evaluation.occ3d_nus_metrics import Metric_mIoU, Metric_FScore
from ..core.evaluation.ray_metrics import calc_ray_iou
from ..core.evaluation.openocc_metric import openocc_metric, format_SC_results, format_SSC_results, cm_to_ious
from torch.utils.data import DataLoader
from .ego_pose_dataset import EgoPoseDataset


colors_map = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
        [0, 0, 0, 0],  # 16 undefined
        [0, 0, 0, 0],  # 16 undefined
    ])


# TODO: [yz] merge to NuScenesDataset
@DATASETS.register_module()
class NuScenesDatasetOccupancy(NuScenesDataset):
    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        resample=True,
        data_type='occ3d'
    ) -> None:
        super().__init__(
            ann_file,
            pipeline,
            dataset_root,
            object_classes,
            map_classes,
            load_interval,
            with_velocity,
            modality,
            box_type_3d,
            filter_empty_gt,
            test_mode,
            eval_version,
            use_valid_flag,
        )
        self.resample = resample
        assert data_type in ['occ3d', 'surround_occ', 'open_occ']
        self.data_type = data_type

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        input_dict = super(NuScenesDatasetOccupancy, self).get_data_info(index)
        # standard protocol modified from SECOND.Pytorch
        # input_dict['occ_gt_path'] = os.path.join(self.data_root, self.data_infos[index]['occ_path'])
        input_dict[self.data_type] = {'occ_gt_path': self.data_infos[index][self.data_type]['occ_path']}
        return input_dict

    def evaluate_occ(self, occ_results, show_dir=None, **eval_kwargs):
        # occ_results: [{'occ_pred': [ndarray:(200, 200, 16)]}, 'occ_score': ] * 6019
        # if isinstance(occ_results, list) and isinstance(occ_results[0], dict):
        #     assert len(occ_results[0]['occ_pred']) == 1
        #     occ_results = [o['occ_pred'][0] for o in occ_results]  # [ndarray:(200, 200, 16)] * 6019
        occ_results = [o['occ_pred'] for o in occ_results]  # [ndarray:(200,200,16)] * 6019
        # metric = eval_kwargs['metric'][0]
        metric = 'mIoU'
        # metric = 'ray-iou'
        if 'occ_metric' in eval_kwargs:
            metric = eval_kwargs['occ_metric']
        assert metric in ['mIoU', 'Ray-IoU']
        print("\nocc metric = ", metric)
        if metric == 'Ray-IoU':
            if self.data_type == 'occ3d':
                occ_gts = []
                occ_preds = []
                lidar_origins = []

                print('\nStarting Evaluation...')

                data_loader = DataLoader(
                    EgoPoseDataset(self.data_infos),
                    batch_size=1,
                    shuffle=False,
                    num_workers=8
                )

                sample_tokens = [info['token'] for info in self.data_infos]

                for i, batch in enumerate(data_loader):
                    token = batch[0][0]
                    output_origin = batch[1]

                    data_id = sample_tokens.index(token)
                    info = self.data_infos[data_id]
                    # occ_gt = np.load(os.path.join(self.data_root, info['occ_path'], 'labels.npz'))
                    occ_gt = np.load(os.path.join(info[self.data_type]['occ_path'], 'labels.npz'))
                    gt_semantics = occ_gt['semantics']      # (Dx, Dy, Dz)
                    mask_lidar = occ_gt['mask_lidar'].astype(bool)      # (Dx, Dy, Dz)
                    mask_camera = occ_gt['mask_camera'].astype(bool)    # (Dx, Dy, Dz)
                    occ_pred = occ_results[data_id]     # (Dx, Dy, Dz)

                    lidar_origins.append(output_origin)
                    occ_gts.append(gt_semantics)
                    occ_preds.append(occ_pred)

                eval_results = calc_ray_iou(occ_preds, occ_gts, lidar_origins)
            else:
                raise NotImplementedError
        else:  # mIoU
            if self.data_type == 'occ3d':
                self.occ_eval_metrics = Metric_mIoU(
                    num_classes=18,
                    use_lidar_mask=False,
                    use_image_mask=True)

                print('\nStarting Evaluation...')
                for index, occ_pred in enumerate(tqdm(occ_results)):
                    # occ_pred: (Dx, Dy, Dz)
                    info = self.data_infos[index]
                    # occ_gt = np.load(os.path.join(self.data_root, info['occ_path'], 'labels.npz'))
                    occ_gt = np.load(os.path.join(info[self.data_type]['occ_path'], 'labels.npz'))
                    gt_semantics = occ_gt['semantics']      # (Dx, Dy, Dz)
                    mask_lidar = occ_gt['mask_lidar'].astype(bool)      # (Dx, Dy, Dz)
                    mask_camera = occ_gt['mask_camera'].astype(bool)    # (Dx, Dy, Dz)
                    # occ_pred = occ_pred
                    self.occ_eval_metrics.add_batch(
                        occ_pred,   # (Dx, Dy, Dz)
                        gt_semantics,   # (Dx, Dy, Dz)
                        mask_lidar,     # (Dx, Dy, Dz)
                        mask_camera     # (Dx, Dy, Dz)
                    )

                    # if index % 100 == 0 and show_dir is not None:
                    #     gt_vis = self.vis_occ(gt_semantics)
                    #     pred_vis = self.vis_occ(occ_pred)
                    #     mmcv.imwrite(np.concatenate([gt_vis, pred_vis], axis=1),
                    #                  os.path.join(show_dir + "%d.jpg"%index))

                    if show_dir is not None:
                        mmcv.mkdir_or_exist(show_dir)
                        # scene_name = info['scene_name']
                        scene_name = [tem for tem in info[self.data_type]['occ_path'].split('/') if 'scene-' in tem][0]
                        sample_token = info['token']
                        mmcv.mkdir_or_exist(os.path.join(show_dir, scene_name, sample_token))
                        save_path = os.path.join(show_dir, scene_name, sample_token, 'pred.npz')
                        np.savez_compressed(save_path, pred=occ_pred, gt=occ_gt, sample_token=sample_token)

                eval_results = self.occ_eval_metrics.count_miou_metric()
            elif self.data_type == 'surround_occ':
                print('\nStarting Evaluation...')
                class_num = 17
                class_names = {0: 'IoU', 1: 'barrier', 2: 'bicycle', 3: 'bus', 4: 'car', 5: 'construction_vehicle',
                               6: 'motorcycle', 7: 'pedestrian', 8: 'traffic_cone', 9: 'trailer', 10: 'truck',
                               11: 'driveable_surface', 12: 'other_flat', 13: 'sidewalk', 14: 'terrain', 15: 'manmade',
                               16: 'vegetation'}
                surround_occ_shape = [200, 200, 16]
                eval_results = {}
                results = []
                for index, occ_pred in enumerate(tqdm(occ_results)):
                    # occ_pred: (Dx, Dy, Dz)
                    info = self.data_infos[index]
                    occ = np.load(info[self.data_type]['occ_path'])
                    occ = occ.astype(np.int32)  # TODO: why np.float32 ?

                    # class 0 is 'ignore' class
                    # if self.use_semantic:
                    #     occ[..., 3][occ[..., 3] == 0] = 255
                    # else:
                    #     occ = occ[occ[..., 3] > 0]
                    #     occ[..., 3] = 1
                    gt = np.zeros(surround_occ_shape, dtype=np.int32)
                    occ[..., 3][occ[..., 3] == 0] = 255
                    coords = occ[:, :3].astype(np.int32)
                    gt[coords[:, 0], coords[:, 1], coords[:, 2]] = occ[:, 3]

                    # gt_i, pred_i = gt_occ[i].cpu().numpy(), pred_occ[i].cpu().numpy()
                    # gt_i = gt_to_voxel(gt_i, img_metas)
                    # pdb.set_trace()  # gt_i: (200, 200, 16), pred_i: (200, 200, 16)
                    mask = (gt != 255)
                    score = np.zeros((class_num, 3))
                    for j in range(class_num):
                        if j == 0:  # class 0 for geometry IoU
                            score[j][0] += ((gt[mask] != 0) * (occ_pred[mask] != 0)).sum()
                            score[j][1] += (gt[mask] != 0).sum()
                            score[j][2] += (occ_pred[mask] != 0).sum()
                        else:
                            score[j][0] += ((gt[mask] == j) * (occ_pred[mask] == j)).sum()
                            score[j][1] += (gt[mask] == j).sum()
                            score[j][2] += (occ_pred[mask] == j).sum()

                    results.append(score)

                results = np.stack(results, axis=0).mean(0)
                mean_ious = []

                for i in range(class_num):
                    tp = results[i, 0]
                    p = results[i, 1]
                    g = results[i, 2]
                    union = p + g - tp
                    mean_ious.append(tp / union)

                for i in range(class_num):
                    eval_results[f'occ/{class_names[i]}/iou'] = mean_ious[i]
                eval_results[f'occ/mIoU'] = np.mean(np.array(mean_ious)[1:])
            else:  # open_occ
                open_occ_shape = [512, 512, 40]
                # init predictions
                SC_metrics = []
                SSC_metrics = []
                for index, occ_pred in enumerate(tqdm(occ_results)):
                    # occ_pred: (Dx, Dy, Dz)
                    info = self.data_infos[index]
                    occ = np.load(info[self.data_type]['occ_path'])
                    occ = occ[:, [2, 1, 0, 3]]  # z y x cls -> x y z cls
                    occ = occ.astype(np.int32)  # TODO: why np.float32 ?

                    # class 0 is 'ignore' class
                    # if self.use_semantic:
                    #     occ[..., 3][occ[..., 3] == 0] = 255
                    # else:
                    #     occ = occ[occ[..., 3] > 0]
                    #     occ[..., 3] = 1
                    gt = np.zeros(open_occ_shape, dtype=np.int32)
                    occ[:, 3][occ[:, 3] == 0] = 255
                    coords = occ[:, :3].astype(np.int32)
                    gt[coords[:, 0], coords[:, 1], coords[:, 2]] = occ[:, 3]

                    SC_metric, _ = openocc_metric(occ_pred, gt, eval_type='SC', empty_idx=0, visible_mask=None)
                    SSC_metric, SSC_occ_metric = openocc_metric(occ_pred, gt, eval_type='SSC', empty_idx=0, visible_mask=None)

                    SC_metrics.append(SC_metric)
                    SSC_metrics.append(SSC_metric)
                eval_results = {}
                res = {}
                res['SC_metric'] = [sum(SC_metrics)]
                res['SSC_metric'] = [sum(SSC_metrics)]

                ''' evaluate SC '''
                evaluation_semantic = sum(res['SC_metric'])  # ndarray: (2, 2)
                ious = cm_to_ious(evaluation_semantic)
                res_table, res_dic = format_SC_results(ious[1:], return_dic=True)
                for key, val in res_dic.items():
                    eval_results['occ/SC_{}'.format(key)] = val

                ''' evaluate SSC '''
                evaluation_semantic = sum(res['SSC_metric'])  # ndarray: (17, 17)
                ious = cm_to_ious(evaluation_semantic)
                res_table, res_dic = format_SSC_results(ious, return_dic=True)
                for key, val in res_dic.items():
                    eval_results['occ/SSC_{}'.format(key)] = val

        return eval_results

    def evaluate(
        self,
        results,
        metric="bbox",
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        **kwargs,
    ):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        metrics = {}

        if "masks_bev" in results[0]:
            metrics.update(self.evaluate_map(results))

        if "boxes_3d" in results[0]:
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

            if isinstance(result_files, dict):
                for name in result_names:
                    print("Evaluating bboxes of {}".format(name))
                    ret_dict = self._evaluate_single(result_files[name])
                metrics.update(ret_dict)
            elif isinstance(result_files, str):
                metrics.update(self._evaluate_single(result_files))

            if tmp_dir is not None:
                tmp_dir.cleanup()

        if "occ_pred" in results[0]:  # results: [{'occ_pred': [ndarray:(200, 200, 16)]}]
            metrics.update(self.evaluate_occ(results, **kwargs))

        return metrics

    def vis_occ(self, semantics):
        # simple visualization of result in BEV
        semantics_valid = np.logical_not(semantics == 17)
        d = np.arange(16).reshape(1, 1, 16)
        d = np.repeat(d, 200, axis=0)
        d = np.repeat(d, 200, axis=1).astype(np.float32)
        d = d * semantics_valid
        selected = np.argmax(d, axis=2)

        selected_torch = torch.from_numpy(selected)
        semantics_torch = torch.from_numpy(semantics)

        occ_bev_torch = torch.gather(semantics_torch, dim=2,
                                     index=selected_torch.unsqueeze(-1))
        occ_bev = occ_bev_torch.numpy()

        occ_bev = occ_bev.flatten().astype(np.int32)
        occ_bev_vis = colors_map[occ_bev].astype(np.uint8)
        occ_bev_vis = occ_bev_vis.reshape(200, 200, 4)[::-1, ::-1, :3]
        occ_bev_vis = cv2.resize(occ_bev_vis,(400,400))
        return occ_bev_vis

