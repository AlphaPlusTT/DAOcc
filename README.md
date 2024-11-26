# DAOcc

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/daocc-3d-object-detection-assisted-multi/prediction-of-occupancy-grid-maps-on-occ3d)](https://paperswithcode.com/sota/prediction-of-occupancy-grid-maps-on-occ3d?p=daocc-3d-object-detection-assisted-multi)

> **DAOcc: 3D Object Detection Assisted Multi-Sensor Fusion for 3D Occupancy Prediction 
> [[`arxiv`](https://arxiv.org/abs/2409.19972)]**
> <br> Zhen Yang, Heng Wang
> <br> Beijing Mechanical Equipment Institute, Beijing, China

This is the official implementation of DAOcc. DAOcc is a novel multi-modal occupancy prediction framework that leverages 3D object detection to assist in achieving superior performance while using a deployment-friendly image encoder and practical input image resolution.

![](figs/overview.jpg)

## News

* **2024-10-01**: Our preprint is available on [arXiv](https://arxiv.org/abs/2409.19972).

## Experimental results

### 3D Semantic Occupancy Prediction on [Occ3D-nuScenes](https://github.com/Tsinghua-MARS-Lab/Occ3D)

| Method | Camera <br/> Mask | Image <br/> Backbone | Image <br/> Resolution | mIoU  |   Config    |     Model      |     Log      |
|:------:|:-----------------:|:--------------------:|:----------------------:|:-----:| :---------: |:--------------:|:------------:|
| DAOcc  |         √         |         R50          |        256×704         | 53.82 | [config](x) | [model](x.pth) | [log](x.log) |

| Method | Camera <br/> Mask | Image <br/> Backbone | Image <br/> Resolution | RayIoU |   Config    |     Model      |     Log      |
|:------:|:-----------------:|:--------------------:|:----------------------:|:------:| :---------: |:--------------:|:------------:|
| DAOcc  |         ×         |         R50          |        256×704         |  48.2  | [config](x) | [model](x.pth) | [log](x.log) |

### 3D Semantic Occupancy Prediction on [SurroundOcc](https://github.com/weiyithu/SurroundOcc)

| Method | Image <br/> Backbone | Image <br/> Resolution | IoU  | mIoU |   Config    |     Model      |     Log      |
|:------:|:--------------------:|:----------------------:|:----:|:----:| :---------: |:--------------:|:------------:|
| DAOcc  |         R50          |        256×704         | 45.0 | 30.5 | [config](x) | [model](x.pth) | [log](x.log) |

## Getting Started

## Citation

```bibtex
@article{yang2024daocc,
  title={DAOcc: 3D Object Detection Assisted Multi-Sensor Fusion for 3D Occupancy Prediction},
  author={Yang, Zhen and Dong, Yanpeng and Wang, Heng},
  journal={arXiv preprint arXiv:2409.19972},
  year={2024}
}
```

## Acknowledgements

Many thanks to these excellent open-source projects:

- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
- [BEVFusion](https://github.com/mit-han-lab/bevfusion)
- [FlashOcc](https://github.com/Yzichen/FlashOCC)
