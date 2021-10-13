import sys
from pathlib import Path

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add yolov5/ to path

# from models.yolo import Model
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import time
import argparse
from utils.datasets import create_dataloader, InfiniteDataLoader, check_dataset, LoadImagesAndLabels
from utils.augmentations import letterbox
import yaml
import sys
import os, os.path as osp
import cv2

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu


class KeypointDataset(Dataset):
    def __init__(self, path, img_size=640, augment=False):
        self.img_size = img_size
        self.augment = augment

        with open(path, 'r') as t:
            self.img_files = t.read().strip().splitlines()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]

        # read image
        im, (h0, w0), (h, w) = self.load_image(img_path)

        # load labels
        labels_path = (osp.splitext(img_path)[0] + '.txt').replace('images', 'labels')

        return im, 0

    def load_image(self, img_path):
        im = cv2.imread(img_path)  # BGR
        assert im is not None, 'Image Not Found ' + img_path
        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        im, ratio, pad = letterbox(im, self.img_size, auto=False, scaleup=self.augment)
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized


def _mp_fn(index, opt):
    device = xm.xla_device()

    WORLD_SIZE = xm.xrt_world_size()
    RANK = xm.get_ordinal()

    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    data_dict = check_dataset(opt.data)
    train_path = data_dict['train']

    if opt.fake_data:
        train_dataset_len = 120000  # Roughly the size of COCO dataset.
        train_loader = xu.SampleGenerator(
            data=(torch.zeros(opt.batch_size // WORLD_SIZE, 3, opt.imgsz, opt.imgsz),
                  torch.zeros(opt.batch_size // WORLD_SIZE, dtype=torch.int64),
                  torch.zeros(opt.batch_size // WORLD_SIZE, dtype=torch.int64),
                  torch.zeros(opt.batch_size // WORLD_SIZE, dtype=torch.int64)),
            sample_count=train_dataset_len // opt.batch_size)
    else:
        if opt.mnist:
            train_dataset = datasets.MNIST(
                osp.join('data/datasets/mnist', str(xm.get_ordinal())),
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize((0.1307,), (0.3081,))]))
        else:
            train_dataset = LoadImagesAndLabels(train_path, opt.imgsz, opt.batch_size // WORLD_SIZE,
                                                hyp=hyp, kp_flip=data_dict['kp_flip'])
            # train_dataset = KeypointDataset(train_path, opt.imgsz)

        train_sampler = None
        if WORLD_SIZE > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=WORLD_SIZE,
                rank=RANK,
                shuffle=True)

        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset,
        #     batch_size=opt.batch_size // WORLD_SIZE,
        #     sampler=train_sampler,
        #     drop_last=True,
        #     shuffle=False if train_sampler else True,
        #     num_workers=opt.workers,
        #     collate_fn=LoadImagesAndLabels.collate_fn)

        train_loader = InfiniteDataLoader(
            train_dataset,
            batch_size=opt.batch_size // WORLD_SIZE,
            num_workers=opt.workers,
            sampler=train_sampler,
            collate_fn=None if opt.mnist else LoadImagesAndLabels.collate_fn
            # collate_fn=None
        )

    train_device_loader = pl.MpDeviceLoader(train_loader, device)

    ti = time.time()
    for i, (imgs, *targets) in enumerate(train_device_loader):
        if i == 100:
            break
        xm.master_print(i, imgs.shape)
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
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--fake-data', action='store_true')
    parser.add_argument('--mnist', action='store_true')
    opt = parser.parse_args()

    # data_dict = check_dataset(opt.data)
    # train_path = data_dict['train']
    #
    # ds = KeypointDataset(train_path)
    # for i in range(10):
    #     img = ds.__getitem__(i)
    #     cv2.imshow('', img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     # print(.shape)
    xmp.spawn(_mp_fn, args=(opt,), nprocs=opt.tpu_cores)
