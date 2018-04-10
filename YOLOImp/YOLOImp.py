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
from yad2k.models.modelV3 import yolo_eval as yolo_eval2
import time
from manager import *
from videoManager import *
from convert import _main as convertYolo

sess = K.get_session()

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.) 
#yolo_model = load_model("model_data/yolov3.h5")
yolov3_model = load_model("model_data/yolov3.h5")

yolo_model.summary()
print(yolov3_model.output)

yolo_outputs = yolo_head(yolov3_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
data = scores, boxes, classes

scores, boxes, classes = yolo_eval2(yolov3_model.output, anchors, len(class_names), image_shape)
data = scores, boxes, classes

predictImage(sess, "test8.jpg", data)




class Expando(object):
    pass

ex = Expando()
ex.config_path = "model_data/yolov3.cfg"
ex.weights_path = "model_data/yolov3.weights"
print()
ex.output_path = "model_data/yolov3.h5"

convertYolo(ex)

#video_to_frames("video/input/test1.mp4", "video/output/")

#processImages(sess, "video/output/", "video/output/processed", data)

#framesToVideo("video/output/processed/", "video/output/processed/video/test_processed.avi")


