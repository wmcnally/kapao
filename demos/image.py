import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add kapao/ to path

import torch
import argparse
import yaml
from utils.torch_utils import select_device
from utils.general import check_img_size, scale_coords
from utils.datasets import LoadImages
from models.experimental import attempt_load
from val import run_nms, post_process_batch
import cv2
import os.path as osp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--img-path', default='res/crowdpose_100024.jpg', help='path to image')

    # plotting options
    parser.add_argument('--bbox', action='store_true')
    parser.add_argument('--kp-bbox', action='store_true')
    parser.add_argument('--pose', action='store_true')
    parser.add_argument('--face', action='store_true')
    parser.add_argument('--color-pose', type=int, nargs='+', default=[255, 0, 255], help='pose object color')
    parser.add_argument('--color-kp', type=int, nargs='+', default=[0, 255, 255], help='keypoint object color')
    parser.add_argument('--line-thick', type=int, default=2, help='line thickness')
    parser.add_argument('--kp-size', type=int, default=1, help='keypoint circle size')
    parser.add_argument('--kp-thick', type=int, default=2, help='keypoint circle thickness')

    # model options
    parser.add_argument('--data', type=str, default='data/coco-kp.yaml')
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--weights', default='kapao_l_coco.pt')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--no-kp-dets', action='store_true', help='do not use keypoint objects')
    parser.add_argument('--conf-thres-kp', type=float, default=0.5)
    parser.add_argument('--conf-thres-kp-person', type=float, default=0.2)
    parser.add_argument('--iou-thres-kp', type=float, default=0.45)
    parser.add_argument('--overwrite-tol', type=int, default=25)
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    parser.add_argument('--flips', type=int, nargs='+', default=[-1])

    args = parser.parse_args()

    with open(args.data) as f:
        data = yaml.safe_load(f)  # load data dict

    # add inference settings to data dict
    data['imgsz'] = args.imgsz
    data['conf_thres'] = args.conf_thres
    data['iou_thres'] = args.iou_thres
    data['use_kp_dets'] = not args.no_kp_dets
    data['conf_thres_kp'] = args.conf_thres_kp
    data['iou_thres_kp'] = args.iou_thres_kp
    data['conf_thres_kp_person'] = args.conf_thres_kp_person
    data['overwrite_tol'] = args.overwrite_tol
    data['scales'] = args.scales
    data['flips'] = [None if f == -1 else f for f in args.flips]
    data['count_fused'] = False

    device = select_device(args.device, batch_size=1)
    print('Using device: {}'.format(device))

    model = attempt_load(args.weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(args.imgsz, s=stride)  # check image size
    dataset = LoadImages(args.img_path, img_size=imgsz, stride=stride, auto=True)

    (_, img, im0, _) = next(iter(dataset))
    img = torch.from_numpy(img).to(device)
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    out = model(img, augment=True, kp_flip=data['kp_flip'], scales=data['scales'], flips=data['flips'])[0]
    person_dets, kp_dets = run_nms(data, out)

    if args.bbox:
        bboxes = scale_coords(img.shape[2:], person_dets[0][:, :4], im0.shape[:2]).round().cpu().numpy()
        for x1, y1, x2, y2 in bboxes:
            cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), args.color_pose, thickness=args.line_thick)

    _, poses, _, _, _ = post_process_batch(data, img, [], [[im0.shape[:2]]], person_dets, kp_dets)

    if args.pose:
        for pose in poses:
            if args.face:
                for x, y, c in pose[data['kp_face']]:
                    cv2.circle(im0, (int(x), int(y)), args.kp_size, args.color_pose, args.kp_thick)
            for seg in data['segments'].values():
                pt1 = (int(pose[seg[0], 0]), int(pose[seg[0], 1]))
                pt2 = (int(pose[seg[1], 0]), int(pose[seg[1], 1]))
                cv2.line(im0, pt1, pt2, args.color_pose, args.line_thick)
            if data['use_kp_dets']:
                for x, y, c in pose:
                    if c:
                        cv2.circle(im0, (int(x), int(y)), args.kp_size, args.color_kp, args.kp_thick)

    if args.kp_bbox:
        bboxes = scale_coords(img.shape[2:], kp_dets[0][:, :4], im0.shape[:2]).round().cpu().numpy()
        for x1, y1, x2, y2 in bboxes:
            cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), args.color_kp, thickness=args.line_thick)

    filename = '{}_{}'.format(osp.splitext(osp.split(args.img_path)[-1])[0], osp.splitext(args.weights)[0])
    if args.bbox:
        filename += '_bbox'
    if args.pose:
        filename += '_pose'
    if args.face:
        filename += '_face'
    if args.kp_bbox:
        filename += '_kpbbox'
    if data['use_kp_dets']:
        filename += '_kp_obj'
    filename += '.png'
    cv2.imwrite(filename, im0)

