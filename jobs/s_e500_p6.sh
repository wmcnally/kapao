#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=1-00:00:00
#SBATCH --job-name=s_e500_p6
#SBATCH --output=%x.out
#SBATCH -p compute_full_node

module load anaconda3
source activate yolo-pose
source deactivate yolo-pose
source activate yolo-pose

python -m torch.distributed.launch --nproc_per_node 4 train.py \
--img 1280 \
--batch 128 \
--epochs 500 \
--data data/coco_kp.yaml \
--hyp data/hyps/hyp.kp-p6.yaml \
--val-scales 1 \
--val-flips -1 \
--weights yolov5s6.pt \
--project runs/s_e500_p6 \
--name train \
--workers 128 \
--resume runs/s_e500_p6/train/weights/last.pt
