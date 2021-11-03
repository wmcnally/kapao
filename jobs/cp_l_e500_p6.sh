#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --job-name=cp_l_e500_p6
#SBATCH --output=%x.out
#SBATCH -p compute_full_node

module load anaconda3
source activate yolo-pose
source deactivate yolo-pose
source activate yolo-pose

python train.py \
--img 1280 \
--batch 48 \
--epochs 500 \
--data data/crowdpose.yaml \
--hyp data/hyps/hyp.kp-p6.yaml \
--val-scales 1 \
--val-flips -1 \
--weights yolov5l6.pt \
--project runs/crowdpose/cp_l_e500_p6 \
--name train \
--workers 128 \
--noval \
--resume runs/crowdpose/cp_l_e500_p6/train/weights/last.pt
