# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 22:15:50 2023

@author: ilanb
"""

from ultralytics import YOLO
from PIL import Image
from google.colab import drive
drive.mount('/content/drive')

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data='/content/drive/MyDrive/Project3Data/data/data.yaml', epochs=190, imgsz=900, batch=20)

##################################################################################

model = YOLO('/content/runs/detect/train/weights/best.pt')

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category

# Run batched inference on a list of images
results = model(['/content/drive/MyDrive/Project3Data/data/evaluation/rasppi.jpg'])

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs


# Show the results
for r in results:
    im_array = r.plot(line_width=3, font_size=3)  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('/content/drive/MyDrive/resultsrasppi.jpg')  # save image
    
##################################################################################

model = YOLO('/content/runs/detect/train/weights/best.pt')

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category

# Run batched inference on a list of images
results = model(['/content/drive/MyDrive/Project3Data/data/evaluation/ardmega.jpg'])

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs


# Show the results
for r in results:
    im_array = r.plot(line_width=3, font_size=3)  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('/content/drive/MyDrive/resultsardmega.jpg')  # save image
    

##################################################################################

model = YOLO('/content/runs/detect/train/weights/best.pt')

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category

# Run batched inference on a list of images
results = model(['/content/drive/MyDrive/Project3Data/data/evaluation/arduno.jpg'])

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs


# Show the results
for r in results:
    im_array = r.plot(line_width=2, font_size=2)  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('/content/drive/MyDrive/resultsarduno.jpg')  # save image