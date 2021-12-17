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
import yaml
from tqdm import tqdm
import imageio
from val import run_nms, post_process_batch
import numpy as np
import gdown
import csv

# youtube id, stream tag, start time, end time
# shuffle: yBZ0Y2t0ceo, 135, 34, 42
# flash mob: 2DiQUX11YaY, 136, 188, 196
# red light green light: nrchfeybHmw, 135, 56, 72

TAG_RES = {135: '480p', 136: '720p', 137: '1080p'}
DEMO_BACKUP = {
    'yBZ0Y2t0ceo': ['1XqaKI8-hjmbz97UX9bI6lKxTYj73ztmf', 'yBZ0Y2t0ceo_480p.mp4'],
    '2DiQUX11YaY': ['1E1azSUE5KXHvCCuFvvM6yUjQDmP3EuSx', '2DiQUX11YaY_720p.mp4'],
    'nrchfeybHmw': ['1Q8awNjA6W4gePbWE5cSAu83CwjiwD0_w', 'nrchfeybHmw_480p.mp4']
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # video options
    parser.add_argument('-p', '--video-path', default='', help='path to video file')
    parser.add_argument('--yt-id', default='yBZ0Y2t0ceo', help='youtube url id')
    parser.add_argument('--tag', type=int, default=135, help='stream tag, 137=1080p, 136=720p, 135=480p')
    parser.add_argument('--color', type=int, nargs='+', default=[255, 255, 255], help='pose color')
    parser.add_argument('--face', action='store_true', help='plot face keypoints')
    parser.add_argument('--display', action='store_true', help='display inference results')
    parser.add_argument('--fps-size', type=int, default=1)
    parser.add_argument('--gif', action='store_true', help='create gif')
    parser.add_argument('--gif-size', type=int, nargs='+', default=[480, 270])
    parser.add_argument('--start', type=int, default=34, help='start time (s)')
    parser.add_argument('--end', type=int, default=42, help='end time (s), -1 for remainder of video')
    parser.add_argument('--kp-size', type=int, default=2, help='keypoint circle size')
    parser.add_argument('--kp-thick', type=int, default=2, help='keypoint circle thickness')
    parser.add_argument('--line-thick', type=int, default=3, help='line thickness')
    parser.add_argument('--alpha', type=float, default=0.4, help='pose alpha')
    parser.add_argument('--kp-obj', action='store_true', help='plot keypoint objects only')
    parser.add_argument('--csv', action='store_true', help='write results so csv file')

    # model options
    parser.add_argument('--data', type=str, default='data/coco-kp.yaml')
    parser.add_argument('--imgsz', type=int, default=1024)
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

    video_path = args.video_path
    if not video_path:
        video_path = args.yt_id + '_' + TAG_RES[args.tag] + '.mp4'
        url = 'https://www.youtube.com/watch?v={}'.format(args.yt_id)

        if not osp.isfile(video_path):
            try:
                yt = YouTube(url)
                # [print(s) for s in yt.streams]
                stream = [s for s in yt.streams if s.itag == args.tag][0]
                print('Downloading demo video...')
                stream.download(filename=video_path)
                print('Done.')
            except Exception as e:
                print('Pytube error: {}'.format(e))
                print('We are working on a patch for pytube...')
                if video_path == DEMO_BACKUP[args.yt_id][1]:
                    print('Fetching backup demo video from google drive')
                    gdown.download("https://drive.google.com/uc?id={}".format(DEMO_BACKUP[args.yt_id][0]))
                else:
                    sys.exit()

    device = select_device(args.device, batch_size=1)
    print('Using device: {}'.format(device))

    model = attempt_load(args.weights, map_location=device)  # load FP32 model
    half = args.half & (device.type != 'cpu')
    if half:  # half precision only supported on CUDA
        model.half()
    stride = int(model.stride.max())  # model stride

    imgsz = check_img_size(args.imgsz, s=stride)  # check image size
    dataset = LoadImages(video_path, img_size=imgsz, stride=stride, auto=True)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    cap = dataset.cap
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.end == -1:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - fps * args.start)
    else:
        n = int(fps * (args.end - args.start))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    gif_frames = []
    out_path = '{}_{}_{}'.format(osp.splitext(video_path)[0], osp.splitext(args.weights)[0],
                                 args.device if args.device == 'cpu' else 'gpu')
    if args.csv:
        f = open(out_path + '.csv', 'w')
        csv_writer = csv.writer(f)

    write_video = not args.display and not args.gif
    if write_video:
        writer = cv2.VideoWriter(out_path + '.mp4',
                                 cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    dataset = tqdm(dataset, desc='Running inference', total=n)
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

        # im0[433:455, 626:816] = np.mean(im0[434:454, 626:816], axis=(0, 1))  # remove patch
        im0_copy = im0.copy()

        # DRAW POSES
        csv_row = []
        for j, (bbox, pose) in enumerate(zip(bboxes, poses)):
            x1, y1, x2, y2 = bbox
            cv2.rectangle(im0_copy, (int(x1), int(y1)), (int(x2), int(y2)), args.color, thickness=1)
            if args.csv:
                for x, y, c in pose:
                    csv_row.extend([x, y, c])
            if args.face:
                for x, y, c in pose[data['kp_face']]:
                    if not args.kp_obj or c:
                        cv2.circle(im0_copy, (int(x), int(y)), args.kp_size, args.color, args.kp_thick)
            for seg in data['segments'].values():
                if not args.kp_obj or (pose[seg[0], -1] and pose[seg[1], -1]):
                    pt1 = (int(pose[seg[0], 0]), int(pose[seg[0], 1]))
                    pt2 = (int(pose[seg[1], 0]), int(pose[seg[1], 1]))
                    cv2.line(im0_copy, pt1, pt2, args.color, args.line_thick)
        im0 = cv2.addWeighted(im0, args.alpha, im0_copy, 1 - args.alpha, gamma=0)

        if i == 0:
            t = time_sync() - t0
        else:
            t = time_sync() - t1

        if not args.gif and args.fps_size:
            cv2.putText(im0, '{:.1f} FPS'.format(1 / t), (5 * args.fps_size, 25 * args.fps_size),
                        cv2.FONT_HERSHEY_SIMPLEX, args.fps_size, (255, 255, 255), thickness=2 * args.fps_size)

        if args.gif:
            gif_img = cv2.cvtColor(cv2.resize(im0, dsize=tuple(args.gif_size)), cv2.COLOR_RGB2BGR)
            if args.fps_size:
                cv2.putText(gif_img, '{:.1f} FPS'.format(1 / t), (5 * args.fps_size, 25 * args.fps_size),
                            cv2.FONT_HERSHEY_SIMPLEX, args.fps_size, (255, 255, 255), thickness=2 * args.fps_size)
            gif_frames.append(gif_img)
        elif write_video:
            writer.write(im0)
        else:
            cv2.imshow('', im0)
            cv2.waitKey(1)

        if args.csv:
            csv_writer.writerow(csv_row)

        t1 = time_sync()
        if i == n - 1:
            break

    cv2.destroyAllWindows()
    cap.release()
    if write_video:
        writer.release()

    if args.gif:
        print('Saving GIF...')
        with imageio.get_writer(out_path + '.gif', mode="I", fps=fps) as writer:
            for idx, frame in tqdm(enumerate(gif_frames)):
                writer.append_data(frame)

    if args.csv:
        f.close()


