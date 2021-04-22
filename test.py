#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：building detection 
@File ：test.py
@Author ：zhouhan
@Date ：2021/4/22 5:56 下午 
'''

# -*- coding: utf-8 -*-

import warnings

warnings.filterwarnings("ignore")
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
from mrcnn.config import Config
from datetime import datetime

# 工程根目录
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "data")

# 加载模型
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_crowdai-mapping-challenge_0029.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class ShapesConfig(Config):
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320  # 320
    IMAGE_MAX_DIM = 384  # 384

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 10

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 10

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50


# import train_tongue
# class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # 1 Background + 1 Building
    IMAGE_MAX_DIM = 320
    IMAGE_MIN_DIM = 320


config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'building']


def detection(path, image_type=1):
    # 提取文件名
    image_name = os.path.split(path)[1]
    image_name = os.path.splitext(image_name)[0]
    image = skimage.io.imread(path)
    a = datetime.now()
    # Run detection
    results = model.detect([image], verbose=1)
    print('results:')
    print(results)
    b = datetime.now()
    # Visualize results
    print("time_cost", (b - a).seconds)
    r = results[0]
    save_path = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                            class_names, r['scores'], figsize=(8, 8),image_name=image_name, image_type=image_type)
    return save_path


# detection('data/val/images/000000000000.jpg', 1)