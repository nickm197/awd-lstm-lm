import os

import torch

print("torch:", torch.__version__)
print("Cuda:", torch.backends.cudnn.cuda)
print("CuDNN:", torch.backends.cudnn.version())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'data')

CACHE_DIR = os.path.join(BASE_DIR, 'cache')

CKPT_DIR = os.path.join(BASE_DIR, 'save')

