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

**2. Occ3D-nuScenes**

For Occupancy Prediction task, download (only) the `gts.tar.gz` from [Occ3D-nuScenes](https://github.com/Tsinghua-MARS-Lab/Occ3D) and arrange the folder as:
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

**3. (Optional) SurroundOcc**

Download the generated dense occupancy labels (resolution 200x200x16 with voxel size 0.5m) from [SurroundOcc](https://github.com/weiyithu/SurroundOcc), then unzip and place them in the `data` folder:
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

**4. (Optional) OpenOccupancy**

Download the v0.1 annotation package from [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy), then unzip and place them in the `data` folder:
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
│   ├── nuscenes_occ/ (optional)
│   ├── nuScenes-Occupancy/ (new)
```

**5. Download the generated [train](https://drive.google.com/file/d/10wSBrdVeuZvAIgsq53VhMpAHzdI-thG5/view?usp=sharing)/[val](https://drive.google.com/file/d/1BFs35DG8p5CYiJoU62FP7u1NMZxi0dwn/view?usp=sharing) pickle files and put them in data. Folder structure:**
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
│   │   ├── nuscenes_infos_train_w_3occ.pkl (new)
│   │   ├── nuscenes_infos_val_w_3occ.pkl (new)
│   ├── nuscenes_occ/ (optional)
│   ├── nuScenes-Occupancy/ (optional)
```
## Waymo
TBD