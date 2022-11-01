import os
import cv2
import json
import yaml
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, Any, List

from models.experimental import attempt_load
from val import run_nms, post_process_batch
from utils.general import scale_coords, check_img_size
from utils.datasets import LoadImages

def save_json(
    value,
    file_path,
):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as filename:
        json.dump(value, filename, indent=4)


def load_detector_config(
        kapao_config: str,
        kapao_weights: str,
) -> dict:

    with open(kapao_config) as fin:
        detector_cfg = yaml.load(fin, Loader=yaml.FullLoader)
    detector_cfg['kapao_weights'] = kapao_weights
    detector_cfg['flips'] = [None if f == -1 else f for f in detector_cfg['flips']]

    return detector_cfg


class KapaoDetector:
    def __init__(self, kapao_config: dict, device_id: int = 0):
        self.kapao_config = kapao_config

        self.device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
        self.detector = attempt_load(kapao_config['kapao_weights'], map_location=self.device)

    def predict(self, image) -> List[List[int]]:

        (img_name, img, im0, _) = image
        img = torch.from_numpy(img).to(self.device)
        img = img / 255.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        detector_preds = self.detector(img, augment=True, kp_flip=self.kapao_config['kp_flip'],
                                       scales=self.kapao_config['scales'], flips=self.kapao_config['flips'])[0]
        person_dets, kp_dets = run_nms(self.kapao_config, detector_preds)

        if person_dets[0].size()[0] == 0:
            return []
        
        _, poses, _, _, _ = post_process_batch(self.kapao_config, img, [], [[im0.shape[:2]]], person_dets, kp_dets)

        poses = [pose.tolist() for pose in poses]
        return poses
    
    @property
    def stride(self):
        return int(self.detector.stride.max())


def extract_keypoints(
    src_images_dir: str, 
    detections_dir: str, 
    weights_path: str = None, 
    device_id: int = 0
):
    config_path = Path(__file__).parent / 'detector_config.yaml'
    if weights_path is None:
        weights_path = Path(__file__).parent / 'weights' / 'kapao_l_coco.pt'
    
    kapao_config = load_detector_config(config_path, weights_path)
    detector = KapaoDetector(kapao_config, device_id=device_id)

    imgsz = check_img_size(kapao_config['imgsz'], s=detector.stride)
    dataset = LoadImages(src_images_dir, img_size=imgsz, stride=detector.stride, auto=True)
    
    for image in tqdm(dataset):
        poses = detector.predict(image)
        image_base_name = Path(image[0]).stem
        detection_path = Path(detections_dir) / f'{image_base_name}.json'
        save_json(poses, detection_path)


def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    
    args.add_argument('--src_images_dir', type=str, required=True)
    args.add_argument('--detections_dir', type=str, required=True)
    args.add_argument('--weights_path', type=str, default=None)
    args.add_argument('--device_id', type=int, default=0)
    
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_keypoints(
        src_images_dir=args.src_images_dir,
        detections_dir=args.detections_dir,
        weights_path=args.weights_path,
        device_id=args.device_id,
    )
