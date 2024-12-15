import os
import random
import numpy as np
import torch


def seed_everything(seed: int):
    os.system("export CUBLAS_WORKSPACE_CONFIG=:4096:8")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
