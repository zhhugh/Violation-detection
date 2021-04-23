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
from mrcnn.utils import compute_overlaps_masks
import colorsys

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


# 计算两个mask之间的IOU
def compute_mask_coverage(mask1, mask2):
    mask1 = np.reshape(mask1 > .5, (-1, 1)).astype(np.float32)
    mask2 = np.reshape(mask2 > .5, (-1, 1)).astype(np.float32)
    intersection = np.dot(mask1.T, mask2)
    area = np.sum(mask2, axis=0)
    # area2 = np.sum(mask2, axis=0)
    # union = (area1[:, None] + area2[None:, ]) - intersection
    # iou = intersection / union
    coverage = intersection / area
    return coverage


def union_mask(masks):
    total_mask = np.sum(masks, axis=2)
    return total_mask


def detection(path, image_type=1):
    # 提取文件名
    image_name = os.path.split(path)[1]
    image_name = os.path.splitext(image_name)[0]
    image = skimage.io.imread(path)
    a = datetime.now()
    # Run detection
    results = model.detect([image], verbose=1)
    print('results:')
    # print(results)
    b = datetime.now()
    # Visualize results
    print("time_cost", (b - a).seconds)
    r = results[0]
    image_save_path = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                                  class_names, r['scores'], figsize=(8, 8), image_name=image_name,
                                                  image_type=image_type)
    return image_save_path


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    return colors


violation_confidence = 0.4


def violation_building_detection(base_image_path, new_image_path):
    """

    @param base_image_path: 变化前图片路径
    @param new_image_path: 变化后图片路径
    @return: 变化前识别结果保存路径, 变化后识别结果保存路径
    """
    violation_building_nums = 0
    colors = random_colors(2)
    base_image = skimage.io.imread(base_image_path)
    new_image = skimage.io.imread(new_image_path)

    base_image_name = os.path.split(base_image_path)[1]
    base_image_name = os.path.splitext(base_image_name)[0]

    new_image_name = os.path.split(new_image_path)[1]
    new_image_name = os.path.splitext(new_image_name)[0]

    base_results = model.detect([base_image], verbose=1)
    new_results = model.detect([new_image], verbose=1)

    base_r = base_results[0]
    new_r = new_results[0]

    base_n = base_r['class_ids'].size
    violation_indexes = [0 for i in range(base_n)]
    base_image_save_path = visualize.display_instances(base_image, base_r['rois'], base_r['masks'], base_r['class_ids'],
                                                       class_names, base_r['scores'], figsize=(8, 8),
                                                       image_name=base_image_name,
                                                       image_type=1, violation_indexes=violation_indexes, colors=colors)

    new_n = new_r['class_ids'].size
    violation_indexes = [0 for i in range(new_n)]
    if base_n != new_n:
        total_mask = union_mask(base_r['masks'])
        for i in range(new_n):
            coverage = compute_mask_coverage(total_mask, new_r['masks'][:, :, i])
            print(coverage)
            if coverage < 0.4:
                print("发现疑似违章建筑")
                violation_indexes[i] = 1
                violation_building_nums += 1
    else:
        print("没有发现违章建筑")
    new_image_save_path = visualize.display_instances(new_image, new_r['rois'], new_r['masks'], new_r['class_ids'],
                                                      class_names, new_r['scores'], figsize=(8, 8),
                                                      image_name=new_image_name,
                                                      image_type=2, violation_indexes=violation_indexes, colors=colors)

    return base_image_save_path, new_image_save_path, violation_building_nums
