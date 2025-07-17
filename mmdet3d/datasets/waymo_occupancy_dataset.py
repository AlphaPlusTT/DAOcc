import mmcv
import numpy as np
import os
import tempfile
import torch
from mmcv.utils import print_log
from os import path as osp
# ERROR ROOT at LINE 331, AT line 236 in format_result, we adjust the worker to be really small
from mmdet3d.datasets import DATASETS 
from mmdet3d.core.bbox import Box3DMode, points_cam2img
from mmdet3d.datasets.kitti_dataset import KittiDataset
# from .waymo_let_metric import compute_waymo_let_metric
from ..core.evaluation.occ3d_waymo_metrics import Metric_mIoU, Metric_FScore
from ..core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)
import pickle
import copy
from tqdm import tqdm

@DATASETS.register_module()
class WaymoDatasetOccupancy(KittiDataset):
    """Waymo Dataset.

    This class serves as the API for experiments on the Waymo Dataset.

    Please refer to `<https://waymo.com/open/download/>`_for data downloading.
    It is recommended to symlink the dataset root to $MMDETECTION3D/data and
    organize them as the doc shows.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': box in LiDAR coordinates
            - 'Depth': box in depth coordinates, usually for indoor dataset
            - 'Camera': box in camera coordinates
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list): The range of point cloud used to filter
            invalid predicted boxes. Default: [-85, -85, -5, 85, 85, 5].
    """

    # TODO: wired, the provided 'object_classes' will modify the default value 'CLASSES' in the parent class.
    CLASSES = ('Car', 'Pedestrian', 'Sign', 'Cyclist')

    def __init__(self,
                 dataset_root,
                 ann_file,
                 split,
                 num_views=5,
                 pts_prefix='velodyne',
                 pipeline=None,
                 object_classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 load_interval=1,
                 gt_bin = None,
                 withimage=True,
                 pose_file=None,
                 pcd_limit_range=[-85, -85, -5, 85, 85, 5],
                 img_info_prototype='mmcv',
                 multi_adj_frame_id_cfg=None,
                 ego_cam='CAM_FRONT',
                 occ_gt_data_root = None,
                 use_larger = True,
                 use_infov_mask=True,
                 use_lidar_mask=False,
                 use_camera_mask=True,
                 stereo=False,
                 filter_difficulty=False):
        super().__init__(
            dataset_root=dataset_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=pipeline,
            classes=object_classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            pcd_limit_range=pcd_limit_range)

        with open(pose_file, 'rb') as f:
            pose_all = pickle.load(f)
            self.pose_all = pose_all
        self.withimage = withimage

        self.num_views = num_views
        assert self.num_views <= 5
        # to load a subset, just set the load_interval in the dataset config
        # self.data_infos_full = self.data_infos
        self.load_interval = load_interval
        
        self.data_infos = self.data_infos[::load_interval]

        # possible_error_index = [19404,8685, 11500,7863,7635,15146,1709,14105,26069,14635,2630,14732,13922,18166,6250,11552,8738,11082,31358,21645,27041,21366,8142,1929,322,15659,6368,13766,9776,16388,22907,25448]
        # if not self.test_mode:
        #     self.data_infos = [self.data_infos[index] for index in possible_error_index]
        if hasattr(self, 'flag'):
            self.flag = self.flag[::load_interval]
            # if not self.test_mode:
            #     self.flag = [self.flag[index] for index in possible_error_index]
        if test_mode == True:
            if gt_bin != None:
                self.gt_bin = gt_bin
            # elif load_interval==1 and 'val' in ann_file:
            #     self.gt_bin = 'gt.bin'
            # elif load_interval==5 and 'val' in ann_file:
            #     self.gt_bin = 'gt_subset.bin'
            # elif load_interval==20 and 'train' in ann_file:
            #     self.gt_bin = 'gt_train_subset.bin'
            # else:
            #     assert gt_bin == 'wrong'

        self.img_info_prototype = img_info_prototype
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
        self.ego_cam = ego_cam
        self.stereo = stereo

        for info in tqdm(self.data_infos):
            sample_idx = info['image']['image_idx']
            scene_idx = sample_idx % 1000000 // 1000
            frame_idx = sample_idx % 1000000 % 1000
            info['cams'] = dict()
            info['scene_token'] = scene_idx
            img_filename = os.path.join(self.dataset_root,
                                    info['image']['image_path'])
            if self.modality['use_camera']:
                info['ego2global_rotation'] = np.array(info['pose'])[:3, :3]
                info['ego2global_translation'] = np.array(info['pose'])[:3, 3]
                info['lidar2ego_rotation'] = np.eye(4)[:3, :3]
                info['lidar2ego_translation'] = np.eye(4)[:3, 3]
                cam_pose = self.pose_all[scene_idx][frame_idx]
                for idx_img in range(self.num_views):     
                    pose = cam_pose[idx_img]
                    # lidar2img = pose['intrinsics'] @ np.linalg.inv(pose['sensor2ego'])
                    sensor2ego = pose['sensor2ego']
                    info['cams'][str(idx_img)] = dict()
                    info['cams'][str(idx_img)]['cam_intrinsic'] = pose['intrinsics']
                    info['cams'][str(idx_img)]['sensor2ego_rotation'] = sensor2ego[:3,:3]
                    info['cams'][str(idx_img)]['sensor2ego_translation'] = sensor2ego[:3,3]
                    info['cams'][str(idx_img)]['ego2global_rotation'] = info['ego2global_rotation']
                    info['cams'][str(idx_img)]['ego2global_translation'] = info['ego2global_translation']
                    
                    if idx_img == 2: 
                        info['cams'][str(idx_img)]['data_path'] =  img_filename.replace('image_0', f'image_3')
                    elif idx_img == 3: 
                        info['cams'][str(idx_img)]['data_path'] =  img_filename.replace('image_0', f'image_2')
                    else:
                        info['cams'][str(idx_img)]['data_path'] =  img_filename.replace('image_0', f'image_{idx_img}')


        self.use_larger = use_larger
        self.occ_gt_data_root = occ_gt_data_root

        self.use_infov_mask = use_infov_mask
        self.use_lidar_mask = use_lidar_mask
        self.use_camera_mask = use_camera_mask
        if not self.test_mode:
            self._set_group_flag()
        self.filter_difficulty = filter_difficulty

    def _get_pts_filename(self, idx):
        pts_filename = osp.join(self.root_split, self.pts_prefix,
                                f'{idx:07d}.bin')
        return pts_filename

    def get_data_info(self, index):

        info = self.data_infos[index]
        # print(index)


        sample_idx = info['image']['image_idx']
        scene_idx = sample_idx % 1000000 // 1000
        frame_idx = sample_idx % 1000000 % 1000
        img_filename = os.path.join(self.dataset_root, info['image']['image_path'])

        # TODO: consider use torch.Tensor only
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P0 = info['calib']['P0'].astype(np.float32)
        lidar2img = P0 @ rect @ Trv2c

        pts_filename = self._get_pts_filename(sample_idx)

        input_dict = dict(
            sample_idx = sample_idx,
            pts_filename = pts_filename,
            scene_token = scene_idx,
            img_prefix=None,
        )
        if self.modality['use_camera']:
            if self.img_info_prototype == 'mmcv':
                image_paths = []
                lidar2img_rts = []
                intrinsics_rts = []
                sensor2ego_rts = []
                for idx_img in range(self.num_views):     
                    pose = self.pose_all[scene_idx][frame_idx][idx_img]
                    lidar2img = pose['intrinsics'] @ np.linalg.inv(pose['sensor2ego'])
                    intrinsics = pose['intrinsics']
                    sensor2ego = pose['sensor2ego']

                    if idx_img == 2: 
                        image_paths.append(img_filename.replace('image_0', f'image_3'))
                    elif idx_img == 3: 
                        image_paths.append(img_filename.replace('image_0', f'image_2'))
                    else:
                        image_paths.append(img_filename.replace('image_0', f'image_{idx_img}'))

                    lidar2img_rts.append(lidar2img)
                    intrinsics_rts.append(intrinsics)
                    sensor2ego_rts.append(sensor2ego)

                input_dict['image_paths'] = image_paths
                input_dict['lidar2img'] = lidar2img_rts
                input_dict['cam_intrinsic'] = intrinsics_rts
                input_dict['sensor2ego'] = sensor2ego_rts
            else:
                assert 'bevdet' in self.img_info_prototype
                input_dict.update(dict(curr=info))
                if '4d' in self.img_info_prototype:
                    info_adj_list = self.get_adj_info(info, index)
                    input_dict.update(dict(adjacent=info_adj_list))

        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos
        # input_dict['ann_infos'] = annos

        # Keep consistent with the input of DAOcc
        input_dict['lidar_path'] = input_dict['pts_filename']
        input_dict['camera2ego'] = input_dict['sensor2ego']
        input_dict['lidar2ego'] = np.eye(4)
        input_dict['lidar2camera'] = [np.linalg.inv(m) for m in input_dict['sensor2ego']]
        input_dict['lidar2image'] = input_dict['lidar2img']
        input_dict['camera_intrinsics'] = input_dict['cam_intrinsic']
        input_dict['camera2lidar'] = input_dict['sensor2ego']

        # for debug
        # input_dict['camera2ego'] = [np.eye(2)] * 5
        # input_dict['lidar2ego'] = np.eye(2)
        # input_dict['lidar2camera'] = [np.eye(2)] * 5
        # input_dict['lidar2image'] = [np.eye(2)] * 5
        # input_dict['camera_intrinsics'] = [np.eye(2)] * 5
        # input_dict['camera2lidar'] = [np.eye(2)] * 5

        return input_dict

    def get_adj_info(self, info, index):
        info_adj_list = []
        adj_id_list = list(range(*self.multi_adj_frame_id_cfg))
        if self.stereo:
            assert self.multi_adj_frame_id_cfg[0] == 1
            assert self.multi_adj_frame_id_cfg[2] == 1
            adj_id_list.append(self.multi_adj_frame_id_cfg[1])
        for select_id in adj_id_list:
            select_id = max(index - select_id, 0)
            if not self.data_infos[select_id]['scene_token'] == info[
                    'scene_token']:
                info_adj_list.append(info)
            else:
                info_adj_list.append(self.data_infos[select_id])
        return info_adj_list

    def get_ann_info(self, index):
        # Use index to get the annos, thus the evalhook could also use this api
        # if self.test_mode == True:
        #     info = self.data_infos[index]
        # else: info = self.data_infos_full[index]

        info = self.data_infos[index]
        
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)

        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = self.remove_dontcare(annos)
        if self.filter_difficulty:
            annos = self.filter_by_difficulty(annos)

        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)

        gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
            self.box_mode_3d, np.linalg.inv(rect @ Trv2c))

        gt_bboxes = annos['bbox']

        selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        gt_bboxes = gt_bboxes[selected].astype('float32')
        gt_names = gt_names[selected]
        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_names=gt_names)
        return anns_results

    def evaluate(self, occ_results, metric='mIoU', runner=None,show_dir=None, **eval_kwargs):
        occ_results = [o['occ_pred'] for o in occ_results]
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=16,
            use_infov_mask=self.use_infov_mask,
            use_lidar_mask=self.use_lidar_mask,
            use_image_mask=self.use_camera_mask)

        print('\nStarting Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]

            gt_semantics, mask_lidar, mask_camera, mask_infov = self.get_occ_gt_from_info(info)
            # occ_pred = occ_pred
            self.occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_infov, mask_lidar, mask_camera)

            if index%100==0 and show_dir is not None:
                gt_vis = self.vis_occ(gt_semantics)
                pred_vis = self.vis_occ(occ_pred)
                mmcv.imwrite(np.concatenate([gt_vis, pred_vis], axis=1),
                             os.path.join(show_dir + "%d.jpg"%index))

        return self.occ_eval_metrics.count_miou()

    def get_occ_gt_from_info(self, info):
        sample_idx = info['image']['image_idx']
        pts_filename = self._get_pts_filename(sample_idx)
        basename = os.path.basename(pts_filename)
        seq_name = basename[1:4]
        frame_name = basename[4:7]
        if self.use_larger:
            file_path = os.path.join(self.occ_gt_data_root, seq_name,  '{}_04.npz'.format(frame_name))
        else:
            file_path = os.path.join(self.occ_gt_data_root, seq_name, '{}.npz'.format(frame_name))
        occ_labels = np.load(file_path)
        semantics = occ_labels['voxel_label']
        mask_infov = occ_labels['infov']
        mask_lidar = occ_labels['origin_voxel_state']
        mask_camera = occ_labels['final_voxel_state']
        crop_x = False
        if crop_x:
            w, h, d = semantics.shape
            semantics = semantics[w//2:, :, :]
            mask_infov = mask_infov[w//2:, :, :]
            mask_lidar = mask_lidar[w//2:, :, :]
            mask_camera = mask_camera[w//2:, :, :]

        return semantics, mask_lidar, mask_camera, mask_infov

    def filter_by_difficulty(self, data_dict):
        """
        Filter arrays in dictionary based on 'difficulty' key
        Args:
            data_dict: Dictionary containing multiple arrays, must have 'difficulty' key
        Returns:
            Filtered dictionary
        """
        # Get difficulty array
        difficulty = data_dict['difficulty']
        if len(difficulty) == 0:
            return data_dict
        # Create mask to keep positions where difficulty is not -1
        mask = difficulty != -1

        filtered_dict = {}
        for key, value in data_dict.items():
            assert isinstance(value, np.ndarray)
            # Filter the array
            filtered_dict[key] = value[mask]

        return filtered_dict

