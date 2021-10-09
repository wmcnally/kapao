# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import json
import os, os.path as osp
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression_kp, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_sync
from utils.callbacks import Callbacks

import tempfile
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        conf_thres=0.01,  # confidence threshold
        iou_thres=0.45,  # NMS IoU threshold
        use_kp_dets=True,
        conf_thres_kp=0.5,
        iou_thres_kp=0.45,
        conf_thres_kp_person=0.2,
        overwrite_tol=50,  # pixels for kp det overwrite
        scales=[0.5, 1, 2],
        flips=[None, 3, None],
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        compute_loss=None,
        ):

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check image size

        # Data
        data = check_dataset(data)  # check

    # Half
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    model.eval()
    nc = int(data['nc'])  # number of classes

    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, rect=False,
                                       prefix=colorstr(f'{task}: '), kp_flip=data['kp_flip'])[0]

    seen = 0
    mp, mr, map50, map, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(4, device=device)
    json_dump = []

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc='Processing {} images'.format(task))):
        t_ = time_sync()
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        t = time_sync()
        t0 += t - t_

        # Run model
        out, train_out = model(img, augment=True, kp_flip=data['kp_flip'], scales=scales, flips=flips)  # inference and training outputs
        t1 += time_sync() - t

        # Compute loss
        if train_out:  # only computed if no scale / flipping
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls, kp

        # Run NMS
        t = time_sync()

        if iou_thres == iou_thres_kp and conf_thres_kp > conf_thres:
            # Combined NMS saves ~0.2 ms / image
            dets = non_max_suppression_kp(out, conf_thres, iou_thres, num_coords=data['num_coords'])
            person_dets = [d[d[:, 5] == 0] for d in dets]
            kp_dets = [d[d[:, 4] >= conf_thres_kp] for d in dets]
            kp_dets = [d[d[:, 5] > 0] for d in kp_dets]
        else:
            person_dets = non_max_suppression_kp(out, conf_thres, iou_thres,
                                                 classes=[0],
                                                 num_coords=data['num_coords'])

            kp_dets = non_max_suppression_kp(out, conf_thres_kp, iou_thres_kp,
                                             classes=list(range(1, 1 + len(data['kp_flip']))),
                                             num_coords=data['num_coords'])
        t2 += time_sync() - t

        # Process images in batch
        for si, (pd, kpd) in enumerate(zip(person_dets, kp_dets)):
            seen += 1
            nd = pd.shape[0]
            nkp = kpd.shape[0]
            if nd:
                path, shape = Path(paths[si]), shapes[si][0]
                img_id = int(osp.splitext(osp.split(path)[-1])[0])

                scores = pd[:, 4].cpu().numpy()  # person detection score
                bboxes = scale_coords(img[si].shape[1:], pd[:, :4], shape).round().cpu().numpy()
                poses = scale_coords(img[si].shape[1:], pd[:, -data['num_coords']:], shape).cpu().numpy()
                poses = poses.reshape((nd, -data['num_coords'], 2))
                poses = np.concatenate((poses, np.zeros((nd, poses.shape[1], 1))), axis=-1)

                if use_kp_dets and nkp:
                    mask = scores > conf_thres_kp_person
                    poses_mask = poses[mask]

                    if len(poses_mask):
                        kpd[:, :4] = scale_coords(img[si].shape[1:], kpd[:, :4], shape)
                        kpd = kpd[:, :6].cpu()

                        for x1, y1, x2, y2, conf, cls in kpd:
                            x, y = np.mean((x1, x2)), np.mean((y1, y2))
                            pose_kps = poses_mask[:, int(cls - 1)]
                            dist = np.linalg.norm(pose_kps[:, :2] - np.array([[x, y]]), axis=-1)
                            kp_match = np.argmin(dist)
                            if conf > pose_kps[kp_match, 2] and dist[kp_match] < overwrite_tol:
                                pose_kps[kp_match] = [x, y, conf]
                        poses[mask] = poses_mask

                for i, (bbox, pose, score) in enumerate(zip(bboxes, poses, scores)):
                    json_dump.append({
                        'image_id': img_id,
                        'category_id': 1,
                        'keypoints': pose.reshape(-1).tolist(),
                        'score': float(score)  # person score
                    })

    if len(json_dump):
        if not training:  # save json
            save_dir, weights_name = osp.split(weights[0])
            json_name = '{}_{}'.format(
                osp.splitext(osp.split(task)[-1])[0],
                osp.splitext(weights_name)[0])
            json_name += '_c{}_i{}'.format(conf_thres, iou_thres)
            if use_kp_dets:
                json_name += '_ck{}_ckp{}_ik{}_t{}'.format(conf_thres_kp,
                                                           conf_thres_kp_person,
                                                           iou_thres_kp,
                                                           overwrite_tol)
            json_path = osp.join(save_dir, json_name + '.json')
        else:
            tmp = tempfile.NamedTemporaryFile(mode='w+b')
            json_path = tmp.name

        with open(json_path, 'w') as f:
            json.dump(json_dump, f)

        annot = osp.join(data['path'], 'annotations', 'person_keypoints_val2017.json')
        coco = COCO(annot)
        result = coco.loadRes(json_path)
        eval = COCOeval(coco, result, iouType='keypoints')
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        map, map50 = eval.stats[:2]

        if training:
            tmp.close()

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t2))  # speeds per image
    if not training:
        os.rename(json_path, osp.splitext(json_path)[0] + '_ap{:.4f}.json'.format(map))
        shape = (batch_size, 3, imgsz, imgsz)
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    model.float()  # for training
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), np.zeros(nc), t  # for compatibility with train


def parse_opt():
    parser = argparse.ArgumentParser(prog='val.py')
    parser.add_argument('--data', type=str, default='data/coco_kp.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--use-kp-dets', action='store_true', help='use keypoint bbox detections')
    parser.add_argument('--conf-thres-kp', type=float, default=0.5)
    parser.add_argument('--conf-thres-kp-person', type=float, default=0.2)
    parser.add_argument('--iou-thres-kp', type=float, default=0.45)
    parser.add_argument('--overwrite-tol', type=int, default=50)
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file
    return opt


def main(opt):
    set_logging()
    print(colorstr('val: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(requirements=FILE.parent / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        run(**vars(opt))

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=opt.imgsz, conf_thres=.25, iou_thres=.45,
                save_json=False, plots=False)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=i, conf_thres=opt.conf_thres,
                              iou_thres=opt.iou_thres, save_json=opt.save_json, plots=False)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
