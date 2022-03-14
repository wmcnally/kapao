import argparse
import json
import os, os.path as osp
import sys
from pathlib import Path

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add kapao/ to path

import numpy as np
import torch
from tqdm import tqdm
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.augmentations import letterbox
from utils.general import check_dataset, check_file, check_img_size, \
    non_max_suppression_kp, scale_coords, set_logging, colorstr
from utils.torch_utils import select_device, time_sync
import tempfile
import cv2
import pickle

PAD_COLOR = (114 / 255, 114 / 255, 114 / 255)


def run_nms(data, model_out):
    if data['iou_thres'] == data['iou_thres_kp'] and data['conf_thres_kp'] >= data['conf_thres']:
        # Combined NMS saves ~0.2 ms / image
        dets = non_max_suppression_kp(model_out, data['conf_thres'], data['iou_thres'], num_coords=data['num_coords'])
        person_dets = [d[d[:, 5] == 0] for d in dets]
        kp_dets = [d[d[:, 4] >= data['conf_thres_kp']] for d in dets]
        kp_dets = [d[d[:, 5] > 0] for d in kp_dets]
    else:
        person_dets = non_max_suppression_kp(model_out, data['conf_thres'], data['iou_thres'],
                                             classes=[0],
                                             num_coords=data['num_coords'])

        kp_dets = non_max_suppression_kp(model_out, data['conf_thres_kp'], data['iou_thres_kp'],
                                         classes=list(range(1, 1 + len(data['kp_flip']))),
                                         num_coords=data['num_coords'])
    return person_dets, kp_dets


def post_process_batch(data, imgs, paths, shapes, person_dets, kp_dets,
                       two_stage=False, pad=0, device='cpu', model=None, origins=None):

    batch_bboxes, batch_poses, batch_scores, batch_ids = [], [], [], []
    n_fused = np.zeros(data['num_coords'] // 2)

    if origins is None:  # used only for two-stage inference so set to 0 if None
        origins = [np.array([0, 0, 0]) for _ in range(len(person_dets))]

    # process each image in batch
    for si, (pd, kpd, origin) in enumerate(zip(person_dets, kp_dets, origins)):
        nd = pd.shape[0]
        nkp = kpd.shape[0]

        if nd:
            path, shape = Path(paths[si]) if len(paths) else '', shapes[si][0]
            img_id = int(osp.splitext(osp.split(path)[-1])[0]) if path else si

            # TWO-STAGE INFERENCE (EXPERIMENTAL)
            if two_stage:
                gs = max(int(model.stride.max()), 32)  # grid size (max stride)
                crops, origins, crop_shapes = [], [], []

                for bbox in pd[:, :4].cpu().numpy():
                    x1, y1, x2, y2 = map(int, map(round, bbox))
                    x1, x2 = max(x1, 0), min(x2, data['imgsz'])
                    y1, y2 = max(y1, 0), min(y2, data['imgsz'])
                    h0, w0 = y2 - y1, x2 - x1
                    crop_shapes.append([(h0, w0)])
                    crop = np.transpose(imgs[si][:, y1:y2, x1:x2].cpu().numpy(), (1, 2, 0))
                    crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=PAD_COLOR)  # add padding
                    h0 += 2 * pad
                    w0 += 2 * pad
                    origins = [np.array([x1 - pad, y1 - pad, 0])]
                    crop_pre = letterbox(crop, data['imgsz'], color=PAD_COLOR, stride=gs, auto=False)[0]
                    crop_input = torch.Tensor(np.transpose(np.expand_dims(crop_pre, axis=0), (0, 3, 1, 2))).to(device)

                    out = model(crop_input, augment=True, kp_flip=data['kp_flip'], scales=data['scales'], flips=data['flips'])[0]
                    person_dets, kp_dets = run_nms(data, out)
                    _, poses, scores, img_ids, _ = post_process_batch(
                        data, crop_input, paths, [[(h0, w0)]], person_dets, kp_dets, device=device, origins=origins)

                    # map back to original image
                    if len(poses):
                        poses = np.stack(poses, axis=0)
                        poses = poses[:, :, :2].reshape(poses.shape[0], -1)
                        poses = scale_coords(imgs[si].shape[1:], poses, shape)
                        poses = poses.reshape(poses.shape[0], data['num_coords'] // 2, 2)
                        poses = np.concatenate((poses, np.zeros((poses.shape[0], data['num_coords'] // 2, 1))), axis=-1)
                    poses = [p for p in poses]  # convert back to list

            # SINGLE-STAGE INFERENCE
            else:
                scores = pd[:, 4].cpu().numpy()  # person detection score
                bboxes = scale_coords(imgs[si].shape[1:], pd[:, :4], shape).round().cpu().numpy()
                poses = scale_coords(imgs[si].shape[1:], pd[:, -data['num_coords']:], shape).cpu().numpy()
                poses = poses.reshape((nd, -data['num_coords'], 2))
                poses = np.concatenate((poses, np.zeros((nd, poses.shape[1], 1))), axis=-1)

                if data['use_kp_dets'] and nkp:
                    mask = scores > data['conf_thres_kp_person']
                    poses_mask = poses[mask]

                    if len(poses_mask):
                        kpd[:, :4] = scale_coords(imgs[si].shape[1:], kpd[:, :4], shape)
                        kpd = kpd[:, :6].cpu()

                        for x1, y1, x2, y2, conf, cls in kpd:
                            x, y = np.mean((x1, x2)), np.mean((y1, y2))
                            pose_kps = poses_mask[:, int(cls - 1)]
                            dist = np.linalg.norm(pose_kps[:, :2] - np.array([[x, y]]), axis=-1)
                            kp_match = np.argmin(dist)
                            if conf > pose_kps[kp_match, 2] and dist[kp_match] < data['overwrite_tol']:
                                pose_kps[kp_match] = [x, y, conf]
                                if data['count_fused']:
                                    n_fused[int(cls - 1)] += 1
                        poses[mask] = poses_mask

                poses = [p + origin for p in poses]

            batch_bboxes.extend(bboxes)
            batch_poses.extend(poses)
            batch_scores.extend(scores)
            batch_ids.extend([img_id] * len(scores))

    return batch_bboxes, batch_poses, batch_scores, batch_ids, n_fused


@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=16,  # batch size
        imgsz=1280,  # inference size (pixels)
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.65,  # NMS IoU threshold
        no_kp_dets=False,
        conf_thres_kp=0.2,
        iou_thres_kp=0.25,
        conf_thres_kp_person=0.3,
        overwrite_tol=50,  # pixels for kp det overwrite
        scales=[1],
        flips=[None],
        rect=False,
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        compute_loss=None,
        count_fused=False,
        two_stage=False,
        pad=0,
        save_oks=False,
        json_name=''
        ):

    if two_stage:  # EXPERIMENTAL
        assert batch_size == 1, 'Batch size must be set to 1 for two-stage processing'
        assert conf_thres >= 0.01, 'Confidence threshold must be >= 0.01 for two-stage processing'
        assert not rect, 'Cannot use rectangular inference with two-stage processing'
        assert not half, 'Two-stage processing must use full precision'

    use_kp_dets = not no_kp_dets

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

    # add inference settings to data dict
    data['imgsz'] = imgsz
    data['conf_thres'] = conf_thres
    data['iou_thres'] = iou_thres
    data['use_kp_dets'] = use_kp_dets
    data['conf_thres_kp'] = conf_thres_kp
    data['iou_thres_kp'] = iou_thres_kp
    data['conf_thres_kp_person'] = conf_thres_kp_person
    data['overwrite_tol'] = overwrite_tol
    data['scales'] = scales
    data['flips'] = flips
    data['count_fused'] = count_fused

    is_coco = 'crowdpose' not in data['path']
    if is_coco:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    else:
        from crowdposetools.coco import COCO
        from crowdposetools.cocoeval import COCOeval

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
        dataloader = create_dataloader(data[task], data['labels'], imgsz, batch_size, gs, rect=rect,
                                       prefix=colorstr(f'{task}: '), kp_flip=data['kp_flip'])[0]

    seen = 0
    mp, mr, map50, mAP, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(4, device=device)
    json_dump = []
    n_fused = np.zeros(data['num_coords'] // 2)

    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc='Processing {} images'.format(task))):
        t_ = time_sync()
        imgs = imgs.to(device, non_blocking=True)
        imgs = imgs.half() if half else imgs.float()  # uint8 to fp16/32
        imgs /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = imgs.shape  # batch size, channels, height, width
        t = time_sync()
        t0 += t - t_

        # Run model
        out, train_out = model(imgs, augment=True, kp_flip=data['kp_flip'], scales=data['scales'], flips=data['flips'])
        t1 += time_sync() - t

        # Compute loss
        if train_out:  # only computed if no scale / flipping
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls, kp

        t = time_sync()

        # NMS
        person_dets, kp_dets = run_nms(data, out)

        # Fuse keypoint and pose detections
        _, poses, scores, img_ids, n_fused_batch = post_process_batch(
            data, imgs, paths, shapes, person_dets, kp_dets, two_stage, pad, device, model)

        t2 += time_sync() - t
        seen += len(imgs)
        if count_fused:
            n_fused += n_fused_batch

        for i, (pose, score, img_id) in enumerate(zip(poses, scores, img_ids)):
            json_dump.append({
                'image_id': img_id,
                'category_id': 1,
                'keypoints': pose.reshape(-1).tolist(),
                'score': float(score)  # person score
            })

    if not training:  # save json
        save_dir, weights_name = osp.split(weights)
        if not json_name:
            json_name = '{}_{}_c{}_i{}_ck{}_ik{}_ckp{}_t{}.json'.format(
                task, osp.splitext(weights_name)[0],
                conf_thres, iou_thres, conf_thres_kp, iou_thres_kp,
                conf_thres_kp_person, overwrite_tol
            )
        else:
            if not json_name.endswith('.json'):
                json_name += '.json'
        json_path = osp.join(save_dir, json_name)
    else:
        tmp = tempfile.NamedTemporaryFile(mode='w+b')
        json_path = tmp.name

    with open(json_path, 'w') as f:
        json.dump(json_dump, f)

    if task in ('train', 'val'):
        annot = osp.join(data['path'], data['{}_annotations'.format(task)])
        coco = COCO(annot)
        result = coco.loadRes(json_path)
        eval = COCOeval(coco, result, iouType='keypoints')
        if 'oks_sigmas' in data:
            eval.params.kpt_oks_sigmas = data['oks_sigmas']
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        mAP, map50 = eval.stats[:2]
        if save_oks:
            oks_path = osp.splitext(json_path)[0] + '_oks.pkl'
            with open(oks_path, 'wb') as f:
                pickle.dump(eval.ious, f)

    if training:
        tmp.close()

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t2))  # speeds per image
    if not training and task != 'test':
        os.rename(json_path, osp.splitext(json_path)[0] + '_ap{:.4f}.json'.format(mAP))
        shape = (batch_size, 3, imgsz, imgsz)
        print(f'Speed: %.3fms pre-process, %.3fms inference, %.3fms NMS per image at shape {shape}' % t)
        if count_fused:
            print('Keypoint Objects Fused:', n_fused)
    model.float()  # for training
    return (mp, mr, map50, mAP, *(loss.cpu() / len(dataloader)).tolist()), np.zeros(nc), t  # for compatibility with train


def parse_opt():
    parser = argparse.ArgumentParser(prog='val.py')
    parser.add_argument('--data', type=str, default='data/coco-kp.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', default='kapao_s_coco.pt')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--imgsz', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--task', default='val', help='train, val, test')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='NMS IoU threshold')
    parser.add_argument('--no-kp-dets', action='store_true', help='do not use keypoint objects')
    parser.add_argument('--conf-thres-kp', type=float, default=0.2)
    parser.add_argument('--conf-thres-kp-person', type=float, default=0.3)
    parser.add_argument('--iou-thres-kp', type=float, default=0.25)
    parser.add_argument('--overwrite-tol', type=int, default=50)
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    parser.add_argument('--flips', type=int, nargs='+', default=[-1])
    parser.add_argument('--rect', action='store_true', help='rectangular input image')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--count-fused', action='store_true', help='count the number of fused keypoint objects')
    parser.add_argument('--two-stage', action='store_true', help='use two-stage inference (experimental)')
    parser.add_argument('--pad', type=int, default=0, help='padding for two-stage inference')
    parser.add_argument('--json-name', type=str, default='', help='optional name for saved json file')
    parser.add_argument('--save-oks', action='store_true', help='save oks scores for all detections (pickle)')
    opt = parser.parse_args()
    opt.flips = [None if f == -1 else f for f in opt.flips]
    opt.data = check_file(opt.data)  # check file
    return opt


def main(opt):
    set_logging()
    print(colorstr('val: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    if opt.task in ('train', 'val', 'test'):  # run normally
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
