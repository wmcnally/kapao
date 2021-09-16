from pycocotools.coco import COCO
import sys
import os, os.path as osp
import argparse
import numpy as np
import yaml


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='coco_kp.yaml')
    parser.add_argument('--hyp', default='hyp.scratch-kp.yaml')
    parser.add_argument('--no-obj-pose', action='store_true')
    parser.add_argument('--skip-no-kp', action='store_true')
    args = parser.parse_args()

    obj_pose = not args.no_obj_pose

    with open(osp.join('data', args.data), 'rb') as f:
        data = yaml.safe_load(f)

    with open(osp.join('data', 'hyps', args.hyp), 'rb') as f:
        hyp = yaml.safe_load(f)

    for split in ['train', 'val']:
        img_paths = []
        img_txt_path = osp.join(data['path'], '{}2017.txt'.format(split))
        img_txt_path_debug = osp.join(data['path'], '{}2017_debug.txt'.format(split))
        labels_path = osp.join(data['path'], 'labels/{}2017'.format(split))
        assert not osp.isdir(labels_path), 'Labels already generated. Remove or rename existing labels folder.'
        os.makedirs(labels_path, exist_ok=False)
        coco = COCO(osp.join(data['path'], 'annotations/person_keypoints_{}2017.json'.format(split)))

        for count, id in enumerate(coco.anns.keys()):
            a = coco.anns[id]

            if split == 'train':
                if a['iscrowd']:
                    continue
                if args.skip_no_kp and ((np.sum(a['keypoints'][2::3]) == 0) or (a['num_keypoints'] == 0)):
                    continue

            if a['image_id'] in coco.imgs:
                img_info = coco.imgs[a['image_id']]

                img_path = osp.join(data['path'], 'images', '{}2017'.format(split), img_info['file_name'])
                if img_path not in img_paths:
                    img_paths.append(img_path)

                img_h, img_w = img_info['height'], img_info['width']
                x, y, w, h = a['bbox']
                xc, yc = x + w/2, y + h/2
                xc /= img_w
                yc /= img_h
                w /= img_w
                h /= img_h

                keypoints = np.array(a['keypoints']).reshape([-1, 3])

                with open(osp.join(labels_path, '{}.txt'.format(img_info['file_name'].split('.')[0])), 'a') as f:
                    # write person object
                    s = '{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(0, xc, yc, w, h)
                    if obj_pose:
                        for i, (x, y, v) in enumerate(keypoints):
                            s += ' {:.6f} {:.6f} {:.6f}'.format(x / img_w, y / img_h, v)
                    s += '\n'
                    f.write(s)

                    # write keypoint objects
                    for i, (x, y, v) in enumerate(keypoints):
                        if v:
                            if isinstance(hyp['kp_bbox'], int):
                                kp_bbox = hyp['kp_bbox']
                            else:
                                kp_bbox = hyp['kp_bbox'][i]

                            s = '{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(
                                i + 1, x / img_w, y / img_h,
                                kp_bbox * max(img_h, img_w) / img_w,
                                kp_bbox * max(img_h, img_w) / img_h)

                            if obj_pose:
                                for _ in range(keypoints.shape[0]):
                                    s += ' {:.6f} {:.6f} {:.6f}'.format(0, 0, 0)
                            s += '\n'
                            f.write(s)

            if (count + 1) % 1000 == 0:
                print('{} {}/{}'.format(split, count + 1, len(coco.anns.keys())))

        print('Writing image paths...')
        with open(img_txt_path, 'w') as f:
            for img_path in img_paths:
                f.write(img_path + '\n')

        with open(img_txt_path_debug, 'w') as f:
            for i, img_path in enumerate(img_paths):
                if i == 128:
                    break
                f.write(img_path + '\n')

