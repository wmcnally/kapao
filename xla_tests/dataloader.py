import sys
from pathlib import Path

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add yolov5/ to path

from models.yolo import Model
import torch
import time
import argparse
from utils.datasets import create_dataloader, check_dataset

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp


def _mp_fn(index):
    device = xm.xla_device()
    data_dict = check_dataset(opt.data)
    train_path = data_dict['train']
    print(train_path)

    # train_loader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
    #                                           hyp=hyp, augment=True, cache=opt.cache, rect=opt.rect, rank=RANK,
    #                                           workers=workers, image_weights=opt.image_weights, quad=opt.quad,
    #                                           prefix=colorstr('train: '), kp_flip=kp_flip)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco_kp.yaml', help='dataset.yaml path')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    opt = parser.parse_args()

    xmp.spawn(_mp_fn, args=(opt,), nprocs=opt.workers)