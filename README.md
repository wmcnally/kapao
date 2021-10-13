# YOLOPose

The official PyTorch implementation for the paper [Modeling Keypoints and Objects as Poses for Bottom-up Human Pose Estimation]().

This repository was forked from ultralytics/yolov5 at commit [5487451](https://github.com/ultralytics/yolov5/tree/5487451)

## Setup
1. If you haven't already, [install Anaconda or Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
2. Create a new conda environment with Python 3.6: `$ conda create -n yolo-pose python=3.6`. Note that newer versions of Python are incompatible with the [CrowdPose API](https://github.com/Jeff-sjtu/CrowdPose/tree/master/crowdpose-api).
3. Activate the environment: `$ conda activate yolo-pose`
4. Clone this repo: `$ git clone https://github.com/wmcnally/yolo-pose.git`
5. Install the dependencies: `$ cd yolo-pose && pip install -r requirements.txt`
6. Install the [CrowdPose API](https://github.com/Jeff-sjtu/CrowdPose/tree/master/crowdpose-api): <br>```$ cd ../ && git clone https://github.com/Jeff-sjtu/CrowdPose.git``` <br> ```$ cd CrowdPose/crowdpose-api/PythonAPI && sh install.sh && cd -```
7. Download the COCO dataset:  `$ cd yolo-pose && sh data/scripts/get_coco_kp.sh`
8. Generate the COCO dataset labels:  `$ python write_kp_labels.py`
9. Download the CrowdPose dataset and place in `data/datasets/crowdpose/` ([images](https://drive.google.com/file/d/1VprytECcLtU4tKP32SYi_7oDRbw7yUTL/view) and [annotations](https://drive.google.com/drive/folders/1Ch1Cobe-6byB7sLhy8XRzOGCGTW2ssFv?usp=sharing)).
10. Generate the CrowdPose dataset labels: `$ python write_kp_labels.py --data crowdpose.yaml`