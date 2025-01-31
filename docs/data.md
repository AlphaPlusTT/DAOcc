# Dataset Preparation

## NuScenes
**1. Download nuScenes V1.0 full dataset data [HERE](https://www.nuscenes.org/download). Folder structure:**
```
DAOcc
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/ (optional)
│   │   ├── v1.0-trainval/
```

*2. Occ3D-nuScenes*

**For Occupancy Prediction task, download (only) the `gts.tar.gz` from [Occ3D-nuScenes](https://github.com/Tsinghua-MARS-Lab/Occ3D) and arrange the folder as:**
```
DAOcc
├── data/
│   ├── nuscenes/
│   │   ├── gts/ (new)
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
```

*3. (Optional) SurroundOcc*

**Download the generated dense occupancy labels (resolution 200x200x16 with voxel size 0.5m) from [SurroundOcc](https://github.com/weiyithu/SurroundOcc), then unzip and place them in the `data` folder:**
```
DAOcc
├── data/
│   ├── nuscenes/
│   │   ├── gts/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
│   ├── nuscenes_occ/ (new)
```

**4. Download the generated [train](https://drive.google.com/file/d/1AyI7Wla482yF_OZUr6rx-qrHi9tXwsPl/view?usp=drive_link)/[val](https://drive.google.com/file/d/1XsCgLQ8bs0jYQQX3GQ9kL5FuZ-_pBSvM/view?usp=drive_link) pickle files and put them in data. Folder structure:**
```
DAOcc
├── data/
│   ├── nuscenes/
│   │   ├── gts/ (new)
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/ (optional)
│   │   ├── v1.0-trainval/
│   │   ├── nuscenes_infos_train_w_2occ.pkl (new)
│   │   ├── nuscenes_infos_val_w_2occ.pkl (new)
│   ├── nuscenes_occ/ (optional)
```
