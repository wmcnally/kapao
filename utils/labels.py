import os, os.path as osp
import argparse
import numpy as np
import yaml
from tqdm import tqdm


def write_kp_labels(data):
    assert not osp.isdir(osp.join(data['path'], data['labels'])), \
        'Labels already generated. Remove or choose new name for labels.'

    is_coco = 'crowdpose' not in data['path']
    if is_coco:
        from pycocotools.coco import COCO
    else:
        from crowdposetools.coco import COCO

    splits = [osp.splitext(osp.split(data[s])[-1])[0] for s in ['train', 'val', 'test'] if s in data]
    annotations = [osp.join(data['path'], data['{}_annotations'.format(s)]) for s in ['train', 'val', 'test'] if s in data]
    test_split = [0 if s in ['train', 'val'] else 1 for s in ['train', 'val', 'test'] if s in data]
    img_txt_dir = osp.join(data['path'], data['labels'], 'img_txt')
    os.makedirs(img_txt_dir, exist_ok=True)

    for split, annot, is_test in zip(splits, annotations, test_split):
        img_txt_path = osp.join(img_txt_dir, '{}.txt'.format(split))
        labels_path = osp.join(data['path'], '{}/{}'.format(data['labels'], split if is_coco else ''))
        if not is_test:
            os.makedirs(labels_path, exist_ok=True)
        coco = COCO(annot)
        if not is_test:
            pbar = tqdm(coco.anns.keys(), total=len(coco.anns.keys()))
            pbar.desc = 'Writing {} labels to {}'.format(split, labels_path)
            for id in pbar:
                a = coco.anns[id]

                if a['image_id'] not in coco.imgs:
                    continue

                if 'train' in split:
                    if is_coco and a['iscrowd']:
                        continue

                img_info = coco.imgs[a['image_id']]
                img_h, img_w = img_info['height'], img_info['width']
                x, y, w, h = a['bbox']
                xc, yc = x + w / 2, y + h / 2
                xc /= img_w
                yc /= img_h
                w /= img_w
                h /= img_h

                keypoints = np.array(a['keypoints']).reshape([-1, 3])

                # some of crowdpose keypoints are just outside image so clip to image extents
                if not is_coco:
                    keypoints[:, 0] = np.clip(keypoints[:, 0], 0, img_w)
                    keypoints[:, 1] = np.clip(keypoints[:, 1], 0, img_h)

                with open(osp.join(labels_path, '{}.txt'.format(osp.splitext(img_info['file_name'])[0])), 'a') as f:
                    # write person object
                    s = '{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(0, xc, yc, w, h)
                    if data['pose_obj']:
                        for i, (x, y, v) in enumerate(keypoints):
                            s += ' {:.6f} {:.6f} {:.6f}'.format(x / img_w, y / img_h, v)
                    s += '\n'
                    f.write(s)

                    # write keypoint objects
                    for i, (x, y, v) in enumerate(keypoints):
                        if v:
                            if isinstance(data['kp_bbox'], list):
                                kp_bbox = data['kp_bbox'][i]
                            else:
                                kp_bbox = data['kp_bbox']

                            s = '{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(
                                i + 1, x / img_w, y / img_h,
                                kp_bbox * max(img_h, img_w) / img_w,
                                kp_bbox * max(img_h, img_w) / img_h)

                            if data['pose_obj']:
                                for _ in range(keypoints.shape[0]):
                                    s += ' {:.6f} {:.6f} {:.6f}'.format(0, 0, 0)
                            s += '\n'
                            f.write(s)
            pbar.close()

        with open(img_txt_path, 'w') as f:
            for img_info in coco.imgs.values():
                f.write(osp.join(data['path'], 'images',
                                 '{}'.format(split if is_coco else ''),
                                 img_info['file_name']) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/coco-kp.yaml')
    args = parser.parse_args()

    assert osp.isfile(args.data), 'Data config file not found at {}'.format(args.data)

    with open(args.data, 'rb') as f:
        data = yaml.safe_load(f)
    write_kp_labels(data)