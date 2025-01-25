## KAIST-datafusion-3Dperception

# INTRODUCTION

The aim of these projects was to create an object detection network capable of identifying different subjects in a real world environment.
The three folders, first_project; second_project; third_project, contain respectively the codes for the thermal object detector, the 
scripts to perform the non-learned data-fusion with the Hungarian algorithm and the programs needed to perfrom a learned data-fusion.

# REQUIREMENTS

Before working on this projects please install all the requirements in a Python 3.9 environment (Note that this might take some time 
and storage space). In order to do that, it is possible to clone this repository and through the 'cd' command from the terminal access to the 
corresponding folder; here, by typing 'sudo pip3 install -r requirements.txt' or  'python -m pip install -r requirements.txt', the downloads 
will start automatically.

# DATASET

The dataset as base for all these projects is the KAIST dataset, that can be found at the following link https://soonminhwang.github.io/rgbt-ped-detection/
Since the downloading page is not working, in order to obtain the full dataset is necessary to follow the instructions reported in the 
"Usage" section.

# FIRST PROJECT

The first project is just a simple object detector training a YOLOv8n on the lwir part of the images in the chosen dataset. The first step is the 
label conversion, that translates the annotation files from the original .xml format into the .txt format required by the YOLO. The code that 
performs this conversion is named "Parser.py", where you need to modify the paths.The second useful code is the "foldercreator.py" which allows to 
reorder all the files from the original structure into a new one; we used this to create the label and the images folders required by the YOLO,
also there is necessay to modify the folder paths. After this it is possible to start the training and the useful codes are "lwirmain.py" and the
"config_lwir.yaml". We reccomand to control the codes to fit your training intension (for example doing a training from scratch or a pretrained
one) and follow the same folder structure we adopted. Lastely there is also the "tester.py" code that is useful to perfrom NN inferences on
validation images and save the results with the bounding boxes drawn on them, also in this code is necessaty to modify all the paths (images and
model).

# SECOND PROJECT

To perform data-fusion we decieded to fuse data coming from the inferences made by the network described in the first project and the datas coming
from an RGB-side network. To obtain the result from the RGB network is possible to follow the same steps described in the first project, but using 
RGB images and the corresponding RGB-related codes "rgbmain.py" and "config_rgb.yaml". At this point there is one code that performs all the
necessary elaboration and the data fusion with the Hungarian algorithm, that code is the "extractor_bbox_definitive.py". This code performs NN
inferences on the images submitted and exploits the Hungarian algorithm to do the actual fusion; at the end the results are saved on a new image
with the bounding box, the class and the confidence shown for every subject.

# THIRD PROJECT

The base of this project was another GitHub repository that can be found at the following link https://github.com/jas-nat/yolov3-KAIST
We want to put the focus on the fact that a new network is needed (YOLOv3) as long as a different portion of the dataset; we added some files
and edited some of the original ones in order to make it compile in a Python 3.9 environment. Be sure to have installed all the requirements.
The useful codes to run and that needs to be eventually modified with your specifics (paths, variables, weights, hyperparameters, etc) are the
"train_kaist_multi.py" and the "detect_multi.py". The first code performs the training with also a testing after every epoch and a final validation
step. The second code instead is useful for a further validation step on a different set of images, specified by the right path, since it
performs the NN inference drawing also the bounding boxes, the class and the confidence on top of them.

# AUTHORS

Federico Rovighi - MUNER, EEIV - ADE master degree

Valerio Tiri - MUNER, EEIV - ADE master degree
