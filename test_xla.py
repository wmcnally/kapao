from models.yolo import Model
import torch
import torch_xla.core.xla_model as xm
import time

# device = 'cpu'
device = xm.xla_device()
model = Model(cfg='models/yolov5s.yaml', nc=5).to(device)
img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)

ti = time.time()
y = model(img)
print('{:.4f} s'.format(time.time() - ti))
for out in y:
    print(out.shape)