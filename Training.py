#!/usr/bin/env python
# coding: utf-8
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import time
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from mrcnn.evaluate import build_coco_results, evaluate_coco
from mrcnn.dataset import MappingChallengeDataset

import zipfile
import urllib.request
import shutil


ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# 预训练模型路径
PRETRAINED_MODEL_PATH = os.path.join(ROOT_DIR, "data/mask_rcnn_balloon.h5")
LOGS_DIRECTORY = os.path.join(ROOT_DIR, "logs")


class MappingChallengeConfig(Config):
    """Configuration for training on data in MS COCO format.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "crowdai-mapping-challenge"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # 1 Backgroun + 1 Building

    STEPS_PER_EPOCH = 500
    VALIDATION_STEPS = 50

    IMAGE_MAX_DIM = 320
    IMAGE_MIN_DIM = 320


config = MappingChallengeConfig()
config.display()

model = modellib.MaskRCNN(mode="training", config=config, model_dir=LOGS_DIRECTORY)
# 加载预训练模型
model_path = PRETRAINED_MODEL_PATH
model.load_weights(model_path, by_name=True)

# Load training dataset
dataset_train = MappingChallengeDataset()
dataset_train.load_dataset(dataset_dir=os.path.join("data", "train"), load_small=True)
dataset_train.prepare()

# Load validation dataset
dataset_val = MappingChallengeDataset()
val_coco = dataset_val.load_dataset(dataset_dir=os.path.join("data", "val"), load_small=True, return_coco=True)
dataset_val.prepare()

# Training - Stage 1
# print("Training network heads")
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE,
#             epochs=10,
#             layers='heads')

# Training - Stage 2
# Finetune layers from ResNet stage 4 and up
# print("Fine tune Resnet stage 4 and up")
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE,
#             epochs=10,
#             layers='4+')

# Training - Stage 3
# Fine tune all layers
print("Fine tune all layers")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=30,
            layers='all')
