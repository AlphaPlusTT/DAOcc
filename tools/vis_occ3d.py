import os, sys
import cv2, imageio
import mayavi.mlab as mlab
import numpy as np
import torch


colors = np.array(
    [
        [0, 0, 0, 255],
        [255, 120, 50, 255],  # barrier              orangey
        [255, 192, 203, 255],  # bicycle              pink
        [255, 255, 0, 255],  # bus                  yellow
        [0, 150, 245, 255],  # car                  blue
        [0, 255, 255, 255],  # construction_vehicle cyan
        [200, 180, 0, 255],  # motorcycle           dark orange
        [255, 0, 0, 255],  # pedestrian           red
        [255, 240, 150, 255],  # traffic_cone         light yellow
        [135, 60, 0, 255],  # trailer              brown
        [160, 32, 240, 255],  # truck                purple
        [255, 0, 255, 255],  # driveable_surface    dark pink
        # [175,   0,  75, 255],       # other_flat           dark red
        [139, 137, 137, 255],
        [75, 0, 75, 255],  # sidewalk             dard purple
        [150, 240, 80, 255],  # terrain              light green
        [230, 230, 250, 255],  # manmade              white
        [0, 175, 0, 255],  # vegetation           green
        [0, 0, 255, 255],  # ego car
        [255, 99, 71, 255],
        [0, 191, 255, 255]
    ]
).astype(np.uint8)

voxel_size = 0.4
pc_range = [-40, -40, -1, 40, 40, 5.4]

visual_path = sys.argv[1]
fov_voxels = np.load(visual_path)

if isinstance(fov_voxels, np.lib.npyio.NpzFile):
    semantics = fov_voxels['semantics']
elif isinstance(fov_voxels, np.ndarray):
    semantics = fov_voxels
else:
    raise NotImplementedError

flatten_list = []
for x in range(200):
    for y in range(200):
        for z in range(16):
            if semantics[x, y, z] != 17:
                flatten_list.append([x, y, z, semantics[x, y, z]])
flatten_list_array = np.array(flatten_list)

fov_voxels = flatten_list_array
if fov_voxels.dtype == 'int':
    fov_voxels = fov_voxels.astype('float')
fov_voxels = fov_voxels[fov_voxels[..., 3] > 0]
fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
fov_voxels[:, 0] += pc_range[0]
fov_voxels[:, 1] += pc_range[1]
fov_voxels[:, 2] += pc_range[2]

figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
plt_plot_fov = mlab.points3d(
    fov_voxels[:, 0],
    fov_voxels[:, 1],
    fov_voxels[:, 2],
    fov_voxels[:, 3],
    colormap="viridis",
    scale_factor=voxel_size - 0.05*voxel_size,
    mode="cube",
    opacity=1.0,
    vmin=0,
    vmax=19,
)


plt_plot_fov.glyph.scale_mode = "scale_by_vector"
plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

mlab.view(azimuth=45, elevation=45, distance=180, focalpoint=[0, 0, -10])
mlab.show()
