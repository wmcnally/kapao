from models.yolo import Model
import torch
import time
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xla', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    if args.xla:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print('Using xla')
    else:
        if args.cpu:
            device = 'cpu'
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {}'.format(device))

    model = Model(cfg='models/yolov5s.yaml', nc=5).to(device)
    img = torch.rand(8, 3, 640, 640).to(device)

    ti = time.time()
    y = model(img)
    print('{:.4f} s'.format(time.time() - ti))
    for out in y:
        print(out.shape)