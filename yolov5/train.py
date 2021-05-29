import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from yolov5.utils.autoanchor import check_anchors
from yolov5.utils.datasets import create_dataloader
from yolov5.utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from yolov5.utils.google_utils import attempt_download
from yolov5.utils.loss import ComputeLoss
from yolov5.utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from yolov5.utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from yolov5.utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

def train():
    pass

def main():
    pass

if __name__ == '__main__' :
    main()
    