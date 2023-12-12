import numpy as np
from torch.utils.tensorboard import SummaryWriter

import argparse
import os
import sys
# sys.path.append('../..')
import torch
import time as t
sys.path.append(os.path.join(os.getcwd(), '..'))
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import glob
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from model.point_net import EncoderDecoder
from model.PN_dataset import MultiClassPointCloudDataset, ToTensor
import wandb
from tqdm import tqdm
import torch.nn as nn
import os.path as osp
# from icecream import ic


dataset_dir = glob.glob('./dataset/train/mustard_bottle/*.npy')
# num=torch.randperm(length).item()
# train_dataset_dir = [num]


fig = plt.figure(figsize=(10, 10))
sample=np.load(dataset_dir[4])


ax1 = fig.add_subplot(141, projection="3d")
ax1.scatter(sample[:, 0], sample[:, 1], sample[:, 2])
ax1.set_title("Original Point Cloud_view1")
ax1.view_init(elev=60, azim=90)  # elev表示仰角，azim表示方位角

sample=np.load(dataset_dir[5])


ax2 = fig.add_subplot(142, projection="3d")
ax2.scatter(sample[:, 0], sample[:, 1], sample[:, 2])
ax2.set_title("Original Point Cloud_view2")
ax2.view_init(elev=20, azim=30)  # elev表示仰角，azim表示方位角

sample=np.load(dataset_dir[6])

ax3 = fig.add_subplot(143, projection="3d")
ax3.scatter(sample[:, 0], sample[:, 1], sample[:, 2])
ax3.view_init(elev=40, azim=60)  # elev表示仰角，azim表示方位角

ax3.set_title("Original Point Cloud_view3")
sample=np.load(dataset_dir[7])


ax4 = fig.add_subplot(144, projection="3d")
ax4.scatter(sample[:, 0], sample[:, 1], sample[:, 2])
ax4.view_init(elev=60, azim=90)  # elev表示仰角，azim表示方位角

ax4.set_title("Original Point Cloud_view4")
plt.tight_layout()
plt.show()
plt.savefig(f'./result/vis/original_plot.png')
# plt.close()