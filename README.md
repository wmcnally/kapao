# YOLOPose

**THIS REPOSITORY IS UNDER ACTIVE DEVELOPMENT. USE WITH CAUTION.**

YOLOPose is an efficient bottom-up multi-person human pose estimation model. When not using test-time augmentation,
it is much faster and more accurate than previous methods like DEKR and HigherHRNet. 

This repository contains the official PyTorch implementation for the paper: **Modeling Keypoints and Poses as Objects for Bottom-up Human Pose Estimation** (link coming soon).

Our code was forked from ultralytics/yolov5 at commit [5487451](https://github.com/ultralytics/yolov5/tree/5487451).

### Environment Setup
1. If you haven't already, [install Anaconda or Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
2. Create a new conda environment with Python 3.6: `$ conda create -n yolo-pose python=3.6`. Note that the [CrowdPose API](https://github.com/Jeff-sjtu/CrowdPose/tree/master/crowdpose-api) is incompatible with Python >= 3.6.
3. Activate the environment: `$ conda activate yolo-pose`
4. Clone this repo: `$ git clone https://github.com/wmcnally/yolo-pose.git`
5. Install the dependencies: `$ cd yolo-pose && pip install -r requirements.txt`
6. Download the trained models from [Google Drive](https://drive.google.com/drive/folders/1ziA3-9NwShjYZ2LHEapJmPe96q2sMMKh?usp=sharing) and place in `yolo-pose/`.

### Inference Demo
To display the inference results in real-time: <br> 
`$ python demos/squash.py --weights yolopose_s_coco.pt --imgsz 1280 --display`

Remove the `--display` argument to write an inference video: <br>
`$ python demos/squash.py --weights yolopose_s_coco.pt --imgsz 1280` <br>

## COCO Experiments
Download the COCO dataset:  `$ sh data/scripts/get_coco_kp.sh`

### Validation (without TTA)
- YOLOPose-S (63.0 AP): `$ python val.py --weights yolopose_s_coco.pt --imgsz 1280`
- YOLOPose-M (68.5 AP): `$ python val.py --weights yolopose_m_coco.pt --imgsz 1280`

### Validation (with TTA)
- YOLOPose-S (64.3 AP): `$ python val.py --weights yolopose_s_coco.pt --imgsz 1280 \ `<br>
`--scales 0.8 1 1.2 --flips -1 3 -1` 
- YOLOPose-M (69.6 AP): `$ python val.py --weights yolopose_m_coco.pt --imgsz 1280 \ `<br>
`--scales 0.8 1 1.2 --flips -1 3 -1` 

### Testing
- YOLOPose-S (63.6 AP): `$ python val.py --weights yolopose_s_coco.pt --imgsz 1280 \ `<br>
`--task test --scales 0.8 1 1.2 --flips -1 3 -1` 
- YOLOPose-M (68.6 AP): `$ python val.py --weights yolopose_m_coco.pt --imgsz 1280 \ `<br>
`--task test --scales 0.8 1 1.2 --flips -1 3 -1` 

### Training
The following commands were used to train the YOLOPose models on 4 V100s with 32GB memory each.

YOLOPose-S:
```
python -m torch.distributed.launch --nproc_per_node 4 train.py \
--img 1280 \
--batch 128 \
--epochs 500 \
--data data/coco-kp.yaml \
--hyp data/hyps/hyp.kp-p6.yaml \
--val-scales 1 \
--val-flips -1 \
--weights yolov5s6.pt \
--project runs/s_e500 \
--name train \
--workers 128 \
```

YOLOPose-M:
```
python train.py \
--img 1280 \
--batch 72 \
--epochs 500 \
--data data/coco-kp.yaml \
--hyp data/hyps/hyp.kp-p6.yaml \
--val-scales 1 \
--val-flips -1 \
--weights yolov5m6.pt \
--project runs/m_e500 \
--name train \
--workers 128 \
```

YOLOPose-L:
```
python train.py \
--img 1280 \
--batch 48 \
--epochs 500 \
--data data/coco-kp.yaml \
--hyp data/hyps/hyp.kp-p6.yaml \
--val-scales 1 \
--val-flips -1 \
--weights yolov5l6.pt \
--project runs/l_e500 \
--name train \
--workers 128 \
```

**Note:** [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) is usually recommended but we found training was less stable for YOLOPose-M/L using DDP. We are investigating this issue.


## CrowdPose Experiments

Coming soon...

[comment]: <> (3. Download the CrowdPose dataset and place in `data/datasets/crowdpose/` &#40;[images]&#40;https://drive.google.com/file/d/1VprytECcLtU4tKP32SYi_7oDRbw7yUTL/view&#41; and [annotations]&#40;https://drive.google.com/drive/folders/1Ch1Cobe-6byB7sLhy8XRzOGCGTW2ssFv?usp=sharing&#41;&#41;.)

[comment]: <> (4. Generate the CrowdPose dataset labels: `$ python write_kp_labels.py --data crowdpose.yaml`)