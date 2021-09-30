#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --job-name=m_e500
#SBATCH --output=%x.out
#SBATCH -p compute_full_node

module load anaconda3
source activate yolo-pose
source deactivate yolo-pose
source activate yolo-pose

python -m torch.distributed.launch --nproc_per_node 4 train.py \
--img 640 \
--batch 224 \
--epochs 500 \
--data data/coco_kp.yaml \
--hyp data/hyps/hyp.kp.yaml \
--weights yolov5m.pt \
--project runs/m_e500 \
--name train \
--workers 128
