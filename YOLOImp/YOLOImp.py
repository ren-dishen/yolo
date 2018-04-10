import argparse
import os
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from utilities import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, preprocess_true_boxes, yolo_loss, yolo_body
import time
from manager import *
from videoManager import *

sess = K.get_session()

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.) 
yolo_model = load_model("model_data/yolo.h5")

yolo_model.summary()

yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

predictImage(sess, "test8.jpg")

#video_to_frames("video/input/test1.mp4", "video/output/")

#processImages(sess, "video/output/", "video/output/processed")

#framesToVideo("video/output/processed/", "video/output/processed/video/test_processed.avi")


