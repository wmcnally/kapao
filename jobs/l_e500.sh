#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=1-00:00:00
#SBATCH --job-name=l_e500
#SBATCH --output=%x.out
#SBATCH -p compute_full_node

scancel 200547

module load anaconda3
source activate yolo-pose
source deactivate yolo-pose
source activate yolo-pose

python -m torch.distributed.launch --nproc_per_node 4 train.py \
--img 640 \
--batch 128 \
--epochs 500 \
--data data/coco_kp.yaml \
--hyp data/hyps/hyp.kp.yaml \
--val-scales 0.5 1 2 \
--val-flips -1 3 -1 \
--weights yolov5l.pt \
--project runs/l_e500 \
--name train \
--workers 128 \
--resume runs/l_e500/train/weights/last.pt
