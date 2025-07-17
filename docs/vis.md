# Visualization

First, run the following command:

```bash
python tools/visualize.py PATH_TO_CONFIG --checkpoint PATH_TO_CHECKPOINT
```

This will generate a folder structured like this:

```
viz/
├── camera-0
│   ├── 1531281439800013-*.png
│   └── ...
├── camera-1
│   ├── 1531281439800013-*.png
│   └── ...
├── camera-2
│   └── ...
├── camera-3
│   └── ...
├── camera-4
│   └── ...
├── camera-5
│   └── ...
├── lidar
│   └── ...
└── occ
    ├── 1531281439800013-*.npy
    ├── 1531281439800013-*_gt.npy
    └── ...
```

* Files with suffix `_gt.npy` represent the ground truth.
* Files with suffix `.npy` represent the model predictions.

**Prerequisites**

Before visualizing the occupancy results, please ensure the following dependencies are installed:
```bash
pip install imageio
pip install vtk==9.3.1
pip install mayavi==4.8.2 --no-cache-dir --verbose --no-build-isolation
pip install PyQt5
```

**Occupancy Visualization**

To visualize the predicted occupancy, use the following command:

```bash
python tools/vis_occ3d.py viz/occ/1531281439800013-*.npy
```

* Replace the file name with the actual `.npy` file name. Do not use wildcards (`*`) in the command.