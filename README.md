# How to train an Object Detector using Tensorflow API on Ubuntu 16.04 (GPU)

This repository is a tutorial for how to use TensorFlow's Object Detection API to train an object detection classifier for multiple objects on Ubuntu 16.04. 

(Click the image below to watch video)

[![Watch the video](https://github.com/Khaivdo/How-to-train-an-Object-Detector-using-Tensorflow-API-on-Ubuntu-16.04-GPU/blob/master/doc/Video.png)](https://youtu.be/sFlbxXRum_0)

## 1. Installation ###


### 1.1 Install Tensorflow-gpu ###

Follow the link below to install CUDA, CuDNN and Tensorflow-gpu if they are not yet installed in your computer. The link will help you to install CUDA 9.0, CuDNN 7.5 and Tensorflow-GPU 1.12. (The tutorial intructs you to install CuDNN version 7.0 but I recommend to install CuDNN version 7.5 to get the best performance while training models on your graphic card).

- [Install CUDA and cuDNN for TensorFlow(GPU) on Ubuntu 16.04](https://medium.com/@zhanwenchen/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e)

If you want to use CUDA 10.0 then have a look [here](https://medium.com/@vitali.usau/install-cuda-10-0-cudnn-7-3-and-build-tensorflow-gpu-from-source-on-ubuntu-18-04-3daf720b83fe), CuDNN version is still recommended to be 7.5, Tensorflow-GPU 1.13 is also required. You should only choose only one of these options, as installing both CUDA versions will cause conflicts between your drivers and libraries.


### 1.2 Build anaconda virtual environment ###

If you don't have Anaconda3 in your system, then you can use the link below to get it done.

- [Install Anaconda3 on Ubuntu 16.04](https://www.rosehosting.com/blog/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04/)

We will create a new anaconda virtual environment named "tf-gpu"  which we will use as our default environment to train the object detector. Open a new Terminal window and create the environment by issuing the command:
  
  `conda create --name tf-gpu python=3.6`
  
Then activate your new environment by issuing:

  `source activate tf-gpu`
  

### 1.3 Download main repositories ###

There are a few repositories that you need to download before you can train your own object detector. These repositories provide you TensorFlow object detection framework, detection models pre-trained on large datasets, configure file and everything you need to train your own Object Detection model. 
Before you download them, create a new folder named "ObjectDetection" on your Desktop, then download and unzip these repositories in that ObjectDetection folder. 

- [Tensorflow Object Detection API](https://github.com/tensorflow/models)

(Click "Clone or download" button to download zipped file, then change the name *models_master* to *models*)

- [Tensorflow Object Detection Models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

(In this tutorial, we will use *faster_rcnn_inception_v2_coco* )

- This repository

(You can delete the */doc* folder, it contains the images for this README only)

Your ObjectDetection folder should look like this:

<img src="https://github.com/Khaivdo/How-to-train-an-Object-Detector-using-Tensorflow-API-on-Ubuntu-16.04-GPU/blob/master/doc/MainDirectory.png" height="350" width="550">


If you want to practice training your own object detector based on my dataset, you can leave all the files as they are and skip steps 2.1, 2.2 and 2.3 below. 

If you want to train your own object detector from scratch, you need to delete:

- All the files in */ObjectDetection/data/test* 
- All the files in */ObjectDetection/data/train*

This tutorial will assume that all the files listed above were deleted, and will go on to explain how to generate the files for your own training dataset.


### 1.4 Complete Tensorflow models installation ###

#### 1.4.1 Install dependent packages ####

There are some other python libraries that we also need to install before completing installation process
```
pip install --user Cython
pip install --user contextlib2
pip install --user pillow
pip install --user lxml
pip install --user jupyter
pip install --user matplotlib
pip install --user pandas
pip install --user opencv-python
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

```

The Tensorflow Object Detection API does not depend on ‘pandas’, ‘opencv-python’ or 'pycocotools' libraries, but 'pandas' is used to generate TFRecords, 'opencv-python' is used to work with images, videos, and webcam feeds, whereas 'pycocotools' allows us to use COCO evaluation metrics. 

### 1.4.2 Compile Protobuf ###

Protobufs is used to configure model and training parameters and it must be compiled before using this object detection framework. Change directories to the models/research/ directory and copy and paste the following command into the command line then press Enter:

    # From ObjectDetection/models/research/
    protoc object_detection/protos/*.proto --python_out=.
    
### 1.4.3 Add libraries to PYTHONPATH ###

When running locally, the /ObjectDetection/models/research/ and slim directories should be appended to PYTHONPATH. This can be done by running the following from tensorflow/models/research/:

    # From ObjectDetection/models/research/
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

Note: This command needs to run from **every new terminal** you start. If you meet the error *ModuleNotFoundError: No module named 'object_detection'*, the reason is you probably forget to issue this command. If you wish to avoid running this manually, you can add it as a new line to the end of your ~/.bashrc file, replacing `pwd` with the absolute path of ObjectDetection/models/research on your system.

### 1.4.4 Test your installation ###

You can test that you have correctly installed the Tensorflow Object Detection API by running the following command:

    python object_detection/builders/model_builder_test.py

The result should look similar like this:

<img src="https://github.com/Khaivdo/How-to-train-an-Object-Detector-using-Tensorflow-API-on-Ubuntu-16.04-GPU/blob/master/doc/CompleteInstallation.png" height="130" width="550">

Note: These steps folllow [installation page](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) but I faced some errors relating to dependent packages at later steps so I decided to re-write it.

## 2 Build your dataset
  
 
### 2.1 Gather images ###

  To train an accurate and robust object detection model, you need to gather a dataset of at least 200 images which have a large variations in backgrounds and lightings, random objects involving in images, position and pose of desired objects. I used around 80 images that each category stays on its own and another 50 images with multiple desired categories in the image to train my Object Detector. Using a dataset with a few images can cause your model to misdefine your desired objects or cannot detect your desired objects in different scenes.

  You can gather images of your objects from Google images or your own image albums.
  
  <img src="https://github.com/Khaivdo/How-to-train-an-Object-Detector-using-Tensorflow-API-on-Ubuntu-16.04-GPU/blob/master/doc/images.png" height="480" width="800">
  
  
  After annotating your images, you can randomly divide your data into two different directories :
  
  - data/train (contain 80% of images)
    
  - data/test (contain 20% of images)
  
  If you have a large amount of images (above 50k), this ratio can be different where you want more images for training process 
  and less images for tesing process.
  
  
### 2.2 Annotate images ###
  
  After gathering images, you will need to install [labelImg](https://github.com/tzutalin/labelImg) to annotate them. 
    Once you have labeled and saved each image, there will be an equivalent .xml file for each of them in the same directory.
 
  <img src="https://github.com/Khaivdo/How-to-train-an-Object-Detector-using-Tensorflow-API-on-Ubuntu-16.04-GPU/blob/master/doc/labelImg1.png" height="480" width="800">
 
 
### 2.3 Create label map ###

Label map file is storaged in *ObjectDetection/training* folder, you can change the names and ids to meet your need. ID should always start from 1.

    item {
      id: 1
      name: 'Calculator'
    }
    item {
      id: 2
      name: 'NintendoSwitch'
    }
    item {
      id: 3
      name: 'Basketball'
    }
 
    
### 2.4 Convert labelled images to training data ###
 
 Tensorflow uses TFRecord files as inputs to train your object detector. Therefore, we will need to do two steps here.
 This repo will use uses the xml_to_csv.py and generate_tfrecord.py scripts from 
    [Dat Tran’s Raccoon Detector](https://github.com/datitran/raccoon_dataset) dataset, with some slight modifications. 

 
#### 2.4.1 Convert .xml file to .csv file ####


To convert .xml file to .csv file, type the command:
      
      # From /ObjectDetection
      python xml_to_csv.py
    
These .csv files will then be used to generate .record files which serve as inputs for Tensorflow. 

    
#### 2.4.2 Generate .record file ####
    
- Open the generate_tfrecord.py file then you can update row_labels below to meet your need:

      # TO-DO replace this with label map
      def class_text_to_int(row_label):
          if row_label == 'Calculator':
              return 1
          if row_label == 'NintendoSwitch':
              return 2
          if row_label == 'Basketball':
              return 3
          else:
              None
              
(Note: make sure that your row_labels and their return numbers match up with your label_map names and IDs in step 2.3)

    
- Type commands to generate .record files. They will be saved in *ObjectDetection/data* folder
      
      # From /ObjectDetection
      # Create train data:
      python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record --image_dir=data/train

      # Create test data:
      python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record  --image_dir=data/test
   
    

### 2.5 Configure the Object Detection Training Pipeline ###

Navigate to C:\tensorflow1\models\research\object_detection\samples\configs and copy the faster_rcnn_inception_v2_coco.config file into the *Object_detection\training* directory.Open the .config file and change the number of desired objecst that you have, directory paths to pre-trained model, .record files and your label_map. Num_examples is the number of images that you put in the *ObjectDetection/data/test* directory. 

- *10* |   num_classes: 3
- *107* |  fine_tune_checkpoint: "/home/james/Desktop/ObjectDetection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"
- *122* |  input_path: "/home/james/Desktop/ObjectDetection/data/train.record"
- *124* |  label_map_path: "/home/james/Desktop/ObjectDetection/training/label_map.pbtxt"
- *128* |  num_examples: 64
- *136* |  input_path: "/home/james/Desktop/ObjectDetection/data/test.record"
- *138* |  label_map_path: "/home/james/Desktop/ObjectDetection/training/label_map.pbtxt"

Save the file after the changes have been made. Now the training job is configured and your model is ready to be trained. 

## 3 Train your model

### 3.1 Training

The training process can be started by issuing the command:

      # From /ObjectDetection
      python models/research/object_detection/model_main.py     --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config     --model_dir=training     --num_train_steps=50000     --sample_1_of_n_eval_examples=1     --alsologtostderr



### 3.2 Tensorboard

During the training, you can then monitor the process of the training and evaluation jobs by running [Tensorboard](https://www.tensorflow.org/tensorboard/r1/summaries) on your local machine. Open a new Terminal and issue the command below:

    source activate tf-gpu
    cd ~/Desktop/ObjectDetection/models/research/
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
    cd ~/Desktop/ObjectDetection
    tensorboard --logdir=training

You will need to copy and paste the http link to a browser to open your Tensorboard.

The most important graphs that you should look for are total_loss graph, loss_1 graph and mAP graphs. 


<img src="https://github.com/Khaivdo/How-to-train-an-Object-Detector-using-Tensorflow-API-on-Ubuntu-16.04-GPU/blob/master/doc/Loss_1.png" height="280" width="350">


<img src="https://github.com/Khaivdo/How-to-train-an-Object-Detector-using-Tensorflow-API-on-Ubuntu-16.04-GPU/blob/master/doc/Total_loss.png" height="280" width="350">

As you can see, after step 22k, my loss_1 stayed steadily at around 0.05 and my total_loss started to increase back. Therefore, I stopped my training process and check the [mAP](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173) graphs.


Total mAP (mean average precision):


<img src="https://github.com/Khaivdo/How-to-train-an-Object-Detector-using-Tensorflow-API-on-Ubuntu-16.04-GPU/blob/master/doc/mAPs.png" height="250" width="650">

The mAPs that I got are pretty good, above 93% for different IoU. Now it is time to export our inference graph and see how the model works.


## 4 Export trained inference graph

Copy the ObjectDetection/models/research/object_detection/export_inference_graph.py script and paste it straight into your ObjectDetection directory.

From your Tensorboard graphs, you can decide which model you should export. Change *-training/model.ckpt-50000* in the command below to *training/model.ckpt-XXXXX* where model.ckpt-XXXXX is the model want to export. You can check the *ObjectDetection/training* to see which models are available to export.

    python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-50000 --output_directory trained-inference-graphs/output_inference_graph_v1.pb

## 5 Use your newly trained Object Detector

Test your model using your webcam by issuing:

    python webcam.py
    
If everything is working properly, the object detector will initialize for a few seconds and the result will be displayed. To close your program, press "q" on your "object detection" window and your program will be terminated.

You also can test the newly trained Object Detector on images and videos. The outputs will then be saved in ObjectDetection directory.

<p align="center">
  <img src="https://github.com/Khaivdo/How-to-train-an-Object-Detector-using-Tensorflow-API-on-Ubuntu-16.04-GPU/blob/master/doc/output1.png" height="500" width="500">
</p>

If you encounter errors, please check out the Common errors below. They were errors that I ran in to while setting up my object detection classifier. You are also encouraged to find solutions for errors on Google. There is usually useful information on Stackoverflow or in TensorFlow’s Issues on GitHub.


## 6 Common errors

**CuDNN failed to initialize**


  *UnknownError (see above for traceback): Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.*
    
Make sure you follow the instruction of installing tensorflow-gpu carefully and environment variables LD_LIBRARY_PATH is exported in your *.bashrc* file

    # To open .bashrc file
    nano ~/.bashrc
    
If you installed tensorflow-gpu by conda, then remove it as well because it can confuse the system.

    conda remove -n tf-gpu cudatoolkit
    

**Unable to decode bytes as JPEG, PNG, GIF, or BMP**


*tensorflow.python.framework.errors_impl.InvalidArgumentError: assertion failed: [Unable to decode bytes as JPEG, PNG, GIF, or BMP]*
  
Delete all images that Image Viewer cannot load (and their .xml files if you already annotated them) in your data directories. (Note: labelImg can load some images that your Image Viewer can't). 

<img src="https://github.com/Khaivdo/How-to-train-an-Object-Detector-using-Tensorflow-API-on-Ubuntu-16.04-GPU/blob/master/doc/JPGimage.png" width="200">


**None has type NoneType**


*return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
TypeError: None has type NoneType, but expected one of: int, long*

Make sure label names are matched up between generate_tfrecord file and label_map file.

E.g. (generate_tfrecord) row_label == 'NintendoSwitch'

(label_map) name: 'NintendoSwitch'






