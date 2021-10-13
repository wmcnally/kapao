import sys
from pathlib import Path

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add yolov5/ to path

from models.yolo import Model
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import time
import argparse
from utils.datasets import create_dataloader, InfiniteDataLoader, check_dataset, LoadImagesAndLabels, load_image
from utils.loss import ComputeLoss
from utils.augmentations import letterbox
import yaml
import sys
import os, os.path as osp
import cv2
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from itertools import repeat
from torch.optim import Adam, SGD, lr_scheduler
import torch.nn as nn

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu


def _mp_fn(index, opt):
    device = xm.xla_device()

    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    data_dict = check_dataset(opt.data)
    num_coords = data_dict['num_coords'] if 'num_coords' in data_dict.keys() else 0
    nc = int(data_dict['nc'])  # number of classes

    model = Model('models/yolov5s.yaml', ch=3, nc=nc, anchors=hyp.get('anchors'), num_coords=num_coords).to(device)

    WORLD_SIZE = xm.xrt_world_size()
    RANK = xm.get_ordinal()

    train_sampler = None
    if WORLD_SIZE > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            TRAIN_DATASET,
            num_replicas=WORLD_SIZE,
            rank=RANK,
            shuffle=True)

    train_loader = InfiniteDataLoader(
        TRAIN_DATASET,
        batch_size=opt.batch_size // WORLD_SIZE,
        num_workers=opt.workers,
        sampler=train_sampler,
        collate_fn=LoadImagesAndLabels.collate_fn
    )

    train_device_loader = pl.MpDeviceLoader(train_loader, device)

    optimizer = SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    compute_loss = ComputeLoss(model, num_coords=num_coords)

    model.train()
    ti = time.time()
    for i, (imgs, targets, paths, _) in enumerate(train_device_loader):
        optimizer.zero_grad()
        if i == 10:
            break
        imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        output = model(imgs)  # forward
        loss, loss_items = compute_loss(output, targets.to(device))  # loss scaled by batch_size
        loss.backward()
        xm.optimizer_step(optimizer)
        xm.master_print(i, loss_items)
    tf = time.time()
    xm.master_print('imgs/s = {:.1f}'.format(100 * opt.batch_size / (tf - ti)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco_kp.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.kp.yaml', help='hyperparameters path')
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--tpu-cores', type=int, default=1)
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')

    opt = parser.parse_args()

    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    data_dict = check_dataset(opt.data)
    train_path = data_dict['train']
    kp_flip = data_dict['kp_flip'] if 'kp_flip' in data_dict.keys() else None

    # load dataset globally, cache into VM memory
    TRAIN_DATASET = LoadImagesAndLabels(train_path, opt.imgsz, opt.batch_size // opt.tpu_cores,
                                        augment=True, hyp=hyp, cache_images=True,
                                        kp_flip=kp_flip)

    xmp.spawn(_mp_fn, args=(opt,), nprocs=opt.tpu_cores, start_method='fork')  # fork to save memory
