######## Object Detection for Image #########
# 
# Author: Khai Do
# Date: 9/3/2019

## Some parts of the code is copied from Tensorflow object detection
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb


# Import libraries
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util


# What model
directPath = os.getcwd()
print(directPath)
MODEL_NAME = os.path.join(directPath, 'trained-inference-graphs/output_inference_graph_v1.pb')

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(directPath, 'training/label_map.pbtxt')

# Number of classes to detect
NUM_CLASSES = 3


# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# Detection

IMG_SIZE = (1200,900)
imagefile = os.path.join(directPath, 'image.jpg') 
img_array = cv2.imread(imagefile)  
image_np = cv2.resize(img_array, IMG_SIZE)
frames = 0  
start = time.time()

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detectionsd
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=3)

            # Display output
            cv2.imshow('object detection', image_np)
            
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.imwrite('output.png',image_np)
                cv2.destroyAllWindows()

