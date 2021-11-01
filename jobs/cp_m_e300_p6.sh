#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --job-name=cp_m_e300_p6
#SBATCH --output=%x.out
#SBATCH -p compute_full_node

module load anaconda3
source activate yolo-pose
source deactivate yolo-pose
source activate yolo-pose

python train.py \
--img 1280 \
--batch 72 \
--epochs 300 \
--data data/crowdpose.yaml \
--hyp data/hyps/hyp.kp-p6.yaml \
--val-scales 1 \
--val-flips -1 \
--weights yolov5m6.pt \
--project runs/cp_m_e300_p6 \
--name train \
--workers 128 \
--noval \
