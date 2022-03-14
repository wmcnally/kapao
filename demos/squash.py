import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add kapao/ to path

import argparse
from pytube import YouTube
import os.path as osp
from utils.torch_utils import select_device, time_sync
from utils.general import check_img_size
from utils.datasets import LoadImages
from models.experimental import attempt_load
import torch
import cv2
import numpy as np
import yaml
from tqdm import tqdm
import imageio
from val import run_nms, post_process_batch
import gdown


VIDEO_NAME = 'Squash MegaRally 176 ReDux - Slow Mo Edition.mp4'
URL = 'https://www.youtube.com/watch?v=Dy62-eTNvY4&ab_channel=PSASQUASHTV'

GRAY = (200, 200, 200)
CROWD_THRES = 450  # max bbox size for crowd classification
CROWD_ALPHA = 0.5
CROWD_KP_SIZE = 2
CROWD_KP_THICK = 2
CROWD_SEG_THICK = 2

BLUE = (245, 140, 66)
ORANGE = (66, 140, 245)
PLAYER_ALPHA_BOX = 0.85
PLAYER_ALPHA_POSE = 0.3
PLAYER_KP_SIZE = 4
PLAYER_KP_THICK = 4
PLAYER_SEG_THICK = 4
FPS_TEXT_SIZE = 3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco-kp.yaml')
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--weights', default='kapao_s_coco.pt')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--no-kp-dets', action='store_true', help='do not use keypoint objects')
    parser.add_argument('--conf-thres-kp', type=float, default=0.5)
    parser.add_argument('--conf-thres-kp-person', type=float, default=0.2)
    parser.add_argument('--iou-thres-kp', type=float, default=0.45)
    parser.add_argument('--overwrite-tol', type=int, default=50)
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    parser.add_argument('--flips', type=int, nargs='+', default=[-1])
    parser.add_argument('--display', action='store_true', help='display inference results')
    parser.add_argument('--fps', action='store_true', help='display fps')
    parser.add_argument('--gif', action='store_true', help='create fig')
    parser.add_argument('--start', type=int, default=20, help='start time (s)')
    parser.add_argument('--end', type=int, default=80, help='end time (s)')
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

    if not osp.isfile(VIDEO_NAME):
        try:
            yt = YouTube(URL)
            # [print(s) for s in yt.streams]
            stream = [s for s in yt.streams if s.itag == 137][0]  # 1080p, 25 fps
            print('Downloading squash demo video...')
            stream.download()
            print('Done.')
        except Exception as e:
            print('Pytube error: {}'.format(e))
            print('We are working on a patch for pytube...')
            print('Fetching backup demo video from google drive')
            gdown.download("https://drive.google.com/uc?id=1S46QDdv0pnU98mNXxechvIVK8qOBFlkl")

    device = select_device(args.device, batch_size=1)
    print('Using device: {}'.format(device))

    model = attempt_load(args.weights, map_location=device)  # load FP32 model
    half = args.half & (device.type != 'cpu')
    if half:  # half precision only supported on CUDA
        model.half()
    stride = int(model.stride.max())  # model stride

    imgsz = check_img_size(args.imgsz, s=stride)  # check image size
    dataset = LoadImages('./{}'.format(VIDEO_NAME), img_size=imgsz, stride=stride, auto=True)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    cap = dataset.cap
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(fps * (args.end - args.start))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    gif_frames = []
    video_name = 'squash_inference_{}'.format(osp.splitext(args.weights)[0])

    if not args.display:
        writer = cv2.VideoWriter(video_name + '.mp4',
                                 cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        if not args.fps:  # tqdm might slows down inference
            dataset = tqdm(dataset, desc='Writing inference video', total=n)

    t0 = time_sync()
    for i, (path, img, im0, _) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        out = model(img, augment=True, kp_flip=data['kp_flip'], scales=data['scales'], flips=data['flips'])[0]
        person_dets, kp_dets = run_nms(data, out)
        bboxes, poses, _, _, _ = post_process_batch(data, img, [], [[im0.shape[:2]]], person_dets, kp_dets)

        bboxes = np.array(bboxes)
        poses = np.array(poses)

        im0_copy = im0.copy()
        player_idx = []

        # DRAW CROWD POSES
        for j, (bbox, pose) in enumerate(zip(bboxes, poses)):
            x1, y1, x2, y2 = bbox
            size = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if size < CROWD_THRES:
                cv2.rectangle(im0_copy, (int(x1), int(y1)), (int(x2), int(y2)), GRAY, thickness=2)
                for x, y, _ in pose[:5]:
                    cv2.circle(im0_copy, (int(x), int(y)), CROWD_KP_SIZE, GRAY, CROWD_KP_THICK)
                for seg in data['segments'].values():
                    pt1 = (int(pose[seg[0], 0]), int(pose[seg[0], 1]))
                    pt2 = (int(pose[seg[1], 0]), int(pose[seg[1], 1]))
                    cv2.line(im0_copy, pt1, pt2, GRAY, CROWD_SEG_THICK)
            else:
                player_idx.append(j)
        im0 = cv2.addWeighted(im0, CROWD_ALPHA, im0_copy, 1 - CROWD_ALPHA, gamma=0)

        # DRAW PLAYER POSES
        player_bboxes = bboxes[player_idx][:2]
        player_poses = poses[player_idx][:2]

        def draw_player_poses(im0, missing=-1):
            for j, (bbox, pose, color) in enumerate(zip(
                    player_bboxes[[orange_player, blue_player]],
                    player_poses[[orange_player, blue_player]],
                    [ORANGE, BLUE])):
                if j == missing:
                    continue
                im0_copy = im0.copy()
                x1, y1, x2, y2 = bbox
                cv2.rectangle(im0_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=-1)
                im0 = cv2.addWeighted(im0, PLAYER_ALPHA_BOX, im0_copy, 1 - PLAYER_ALPHA_BOX, gamma=0)
                im0_copy = im0.copy()
                for x, y, _ in pose:
                    cv2.circle(im0_copy, (int(x), int(y)), PLAYER_KP_SIZE, color, PLAYER_KP_THICK)
                for seg in data['segments'].values():
                    pt1 = (int(pose[seg[0], 0]), int(pose[seg[0], 1]))
                    pt2 = (int(pose[seg[1], 0]), int(pose[seg[1], 1]))
                    cv2.line(im0_copy, pt1, pt2, color, PLAYER_SEG_THICK)
                im0 = cv2.addWeighted(im0, PLAYER_ALPHA_POSE, im0_copy, 1 - PLAYER_ALPHA_POSE, gamma=0)
            return im0

        if i == 0:
            # orange player on left at start
            orange_player = np.argmin(player_bboxes[:, 0])
            blue_player = int(not orange_player)
            im0 = draw_player_poses(im0)
        else:
            # simple player tracking based on frame-to-frame pose difference
            dist = []
            for pose in poses_last:
                dist.append(np.mean(np.linalg.norm(player_poses[0, :, :2] - pose[:, :2], axis=-1)))
            if np.argmin(dist) == 0:
                orange_player = 0
            else:
                orange_player = 1
            blue_player = int(not orange_player)

            # if only one player detected, find which player is missing
            missing = -1
            if len(player_poses) == 1:
                if orange_player == 0:  # missing blue player
                    player_poses = np.concatenate((player_poses, poses_last[1:]), axis=0)
                    player_bboxes = np.concatenate((player_bboxes, bboxes_last[1:]), axis=0)
                    missing = 1
                else:  # missing orange player
                    player_poses = np.concatenate((player_poses, poses_last[:1]), axis=0)
                    player_bboxes = np.concatenate((player_bboxes, bboxes_last[:1]), axis=0)
                    missing = 0
            im0 = draw_player_poses(im0, missing)

        bboxes_last = player_bboxes[[orange_player, blue_player]]
        poses_last = player_poses[[orange_player, blue_player]]

        if i == 0:
            t = time_sync() - t0
        else:
            t = time_sync() - t1

        if args.fps:
            s = FPS_TEXT_SIZE
            cv2.putText(im0, '{:.1f} FPS'.format(1 / t), (5*s, 25*s),
                        cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), thickness=2*s)

        if args.gif:
            gif_frames.append(cv2.resize(im0, dsize=None, fx=0.25, fy=0.25)[:, :, [2, 1, 0]])
        elif not args.display:
            writer.write(im0)
        else:
            cv2.imshow('', cv2.resize(im0, dsize=None, fx=0.5, fy=0.5))
            cv2.waitKey(1)

        t1 = time_sync()
        if i == n - 1:
            break

    cv2.destroyAllWindows()
    cap.release()
    if not args.display:
        writer.release()

    if args.gif:
        print('Saving GIF...')
        with imageio.get_writer(video_name + '.gif', mode="I", fps=fps) as writer:
            for idx, frame in tqdm(enumerate(gif_frames)):
                writer.append_data(frame)



