import sys
from pathlib import Path

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add yolov5/ to path

from models.yolo import Model
import torch
import time
import argparse
from utils.datasets import create_dataloader, InfiniteDataLoader, check_dataset, LoadImagesAndLabels
import yaml
import sys

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp


def _mp_fn(index, opt):
    device = xm.xla_device()

    WORLD_SIZE = xm.xrt_world_size()
    RANK = xm.get_ordinal()

    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    data_dict = check_dataset(opt.data)
    train_path = data_dict['train']

    train_dataset = LoadImagesAndLabels(train_path, opt.imgsz, opt.batch_size // WORLD_SIZE,
                                        hyp=hyp, kp_flip=data_dict['kp_flip'])

    train_sampler = None
    if WORLD_SIZE > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=WORLD_SIZE,
            rank=RANK,
            shuffle=True)

    train_loader = InfiniteDataLoader(
        train_dataset,
        batch_size=opt.batch_size // WORLD_SIZE,
        num_workers=opt.workers,
        sampler=train_sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn
    )

    train_device_loader = pl.MpDeviceLoader(train_loader, device)

    for i, (imgs, targets, paths, _) in enumerate(train_device_loader):
        print(imgs.shape)
        break



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco_kp.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.kp.yaml', help='hyperparameters path')
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--tpu-cores', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    opt = parser.parse_args()

    xmp.spawn(_mp_fn, args=(opt,), nprocs=opt.tpu_cores)