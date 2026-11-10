import random
from torchvision import transforms
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import cv2
import os
import time
import warnings
import torch
import torch.nn.functional as F

warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=UserWarning)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from loss import *
from torchvision.utils import save_image




def load_images(path, sort_key=None, img_size=256):
    path_list = os.listdir(path)
    if sort_key:
        path_list.sort(key=sort_key)

    images = []
    for item in path_list:
        impath = os.path.join(path, item)
        img = cv2.imread(impath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        images.append(img)

    return images



def prepare_data(config):
    print("Loading training data...")

    training_x = load_images(config.train_raw_path,
                             sort_key=lambda x: int(x.split('.')[0]))
    training_y = load_images(config.train_gt_path,
                             sort_key=lambda x: int(x.split('.')[0]))
    X_train, Y_train = [], []
    for img, gt in zip(training_x, training_y):
        img = transforms.ToTensor()(img)
        gt = transforms.ToTensor()(gt)
        X_train.append(img)
        Y_train.append(gt)

    X_train = torch.stack(X_train)
    Y_train = torch.stack(Y_train)

    return X_train, Y_train


def process_image(input_path: str, output_path: str, model: nn.Module) -> float:
    imgx = cv2.imread(input_path)
    h, w = imgx.shape[:2]
    pad_h = 0 if h % 2 == 0 else 1
    pad_w = 0 if w % 2 == 0 else 1
    if pad_h > 0 or pad_w > 0:
        imgx = cv2.copyMakeBorder(imgx, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

    imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
    imgx = np.array(imgx).astype(np.float32)
    imgx = torch.from_numpy(imgx).to("cuda")
    imgx = imgx.permute(2, 0, 1).unsqueeze(0)
    imgx = imgx / 255.0
    imgx = imgx.half()

    with torch.inference_mode():
        torch.cuda.synchronize()
        start_time = time.time()
        output = model(imgx)
        torch.cuda.synchronize()
        batch_time = time.time() - start_time
        # print(batch_time)
    # save_image(output.data, output_path, nrow=1, normalize=True)

    return batch_time