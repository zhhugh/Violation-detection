# -*- coding: utf-8 -*-

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from mrcnn.config import Config
# import utils
from mrcnn import model as modellib, utils
from mrcnn import visualize
import yaml
from mrcnn.model import log
from PIL import Image
from mrcnn.dataset import MappingChallengeDataset

# Root directory of the project
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "data/mask_rcnn_balloon.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MAX_DIM = 320
    IMAGE_MIN_DIM = 320

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 50

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 50
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 20


config = ShapesConfig()
config.display()

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax



def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存txt文件成功")



# train与val数据集准备
dataset_train = MappingChallengeDataset()
dataset_train.load_dataset(dataset_dir=os.path.join("data", "train"), load_small=True)
dataset_train.prepare()


dataset_val = MappingChallengeDataset()
val_coco = dataset_val.load_dataset(dataset_dir=os.path.join("data", "val"), load_small=True, return_coco=True)
dataset_val.prepare()


# mAP
# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
# model_path = model.find_last()

model_path = os.path.join("data/mask_rcnn_crowdai-mapping-challenge_0029.h5")  # 修改成自己训练好的模型
# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
    modellib.load_image_gt(dataset_val, inference_config,
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)
#
# visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
#                              dataset_train.class_names, figsize=(8, 8))

# results = model.detect([original_image], verbose=1)
# #
# r = results[0]
# visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
#                              dataset_val.class_names, r['scores'], ax=get_ax())

img_list = np.random.choice(dataset_val.image_ids, 10)
count1 = 0
APs = []
save_box = np.array([])
save_class = np.array([])
save_mask = np.array([])
save_roi = np.array([])
save_id = np.array([])
save_score = np.array([])
save_m = np.array([])

for image_id in img_list:
    # 加载测试集的ground truth
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    # 将所有ground truth载入并保存
    if count1 == 0:
        save_box, save_class, save_mask = gt_bbox, gt_class_id, gt_mask
    else:
        save_box = np.concatenate((save_box, gt_bbox), axis=0)
        save_class = np.concatenate((save_class, gt_class_id), axis=0)
        save_mask = np.concatenate((save_mask, gt_mask), axis=2)

    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)

    # 启动检测
    results = model.detect([image], verbose=0)
    r = results[0]

    # 将所有检测结果保存
    if count1 == 0:
        save_roi, save_id, save_score, save_m = r["rois"], r["class_ids"], r["scores"], r['masks']
    else:
        save_roi = np.concatenate((save_roi, r["rois"]), axis=0)
        save_id = np.concatenate((save_id, r["class_ids"]), axis=0)
        save_score = np.concatenate((save_score, r["scores"]), axis=0)
        save_m = np.concatenate((save_m, r['masks']), axis=2)

    count1 += 1

# 计算AP, precision, recall
AP, precisions, recalls, overlaps = \
    utils.compute_ap(save_box, save_class, save_mask,
                     save_roi, save_id, save_score, save_m)

print("AP: ", AP)
print("mAP: ", np.mean(AP))

plt.plot(recalls, precisions, 'b', label='PR')
plt.title('precision-recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

# 保存precision, recall信息用于后续绘制图像
text_save('Kpreci.txt', precisions)
text_save('Krecall.txt', recalls)