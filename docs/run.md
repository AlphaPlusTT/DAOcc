# Train and Test

**Prerequisites**

Download the pre-trained weight of the image backbone R50 from [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/1.0/configs/nuimages), 
and subsequently remap the keys within the weight. Alternatively, you can directly download the processed weight file [htc_r50_backbone.pth](https://drive.google.com/file/d/19S91tPjfM2laHKipL7QDs5-6m9pQxQGz/view?usp=drive_link).

**Training**

Train DAOcc with 8 RTX4090 GPUs:
```
torchpack dist-run -np 8 python tools/train.py PATH_TO_CONFIG --run-dir CHECKPOINT_SAVE_DIR --dist --model.encoders.camera.backbone.init_cfg.checkpoint PATH_TO_PRETRAIN
```

Train DAOcc with single GPU:
```
python tools/train.py PATH_TO_CONFIG --run-dir CHECKPOINT_SAVE_DIR --model.encoders.camera.backbone.init_cfg.checkpoint PATH_TO_PRETRAIN
```

**Evaluation**

Evaluate DAOcc with 8 RTX4090 GPUs:
```
torchpack dist-run -np 8 python tools/test.py PATH_TO_CONFIG PATH_TO_WEIGHT --dist --eval bbox
```

Evaluate DAOcc with single GPU:
```
python tools/test.py PATH_TO_CONFIG PATH_TO_WEIGHT --eval bbox
```

**Visualization**

```
python tools/visualize.py PATH_TO_CONFIG --checkpoint PATH_TO_CHECKPOINT
```
