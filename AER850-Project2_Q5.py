# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 01:28:26 2023

@author: ilanb
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the previously trained model
model_path = 'C:\\Users\\ilanb\\OneDrive\\Documents\\GitHub\\AER850\\model.h5'  # Replace with the path to your saved model
model = load_model(model_path)
class_indices_path = 'C:\\Users\\ilanb\\OneDrive\\Documents\\GitHub\\AER850\\class_indices.json'
with open(class_indices_path, 'r') as class_file:
    class_indices = json.load(class_file)

# Function to process and predict a single image
def processing(img_path):
    img = image.load_img(img_path, target_size=(100, 100))
    img_array = image.img_to_array(img)  #convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  #add batch dimension
    img_array /= 255.  #rescale by 1/255
    return img_array

def predicting(img_path, model):
    # Load the image file, target size must match the input size of the model
    img_array = processing(img_path)
    stat = model.predict(img_array)
    return stat

test_image_path = {
    'None': 'C:\\Users\\ilanb\\OneDrive\\Documents\\GitHub\\AER850\\Project 2 Data\\Data\\Test\\None\\Crack__20180418_15_51_11,892.bmp',
    'Small': 'C:\\Users\\ilanb\\OneDrive\\Documents\\GitHub\\AER850\\Project 2 Data\\Data\\Test\\Small\\Crack__20180419_00_25_48,570.bmp',
    'Medium': 'C:\\Users\\ilanb\\OneDrive\\Documents\\GitHub\\AER850\\Project 2 Data\\Data\\Test\\Medium\\Crack__20180419_06_16_35,563.bmp',
    'Large': 'C:\\Users\\ilanb\\OneDrive\\Documents\\GitHub\\AER850\\Project 2 Data\\Data\\Test\\Large\\Crack__20180419_13_29_14,846.bmp'   
}

# Plot the image with predictions
for true_label, img_path in test_image_path.items():
    image_name = os.path.basename(img_path)
    print(f"Processing image: {image_name}")  # Print the image name to the console

    probabilities = predicting(img_path, model)[0]
    predicted_label = np.argmax(probabilities)
    predicted_label_name = list(class_indices.keys())[predicted_label]  # Get the predicted label name

    # Load the image
    img = image.load_img(img_path)  # Load image without rescaling for display
    plt.figure(figsize=(6, 8))  # Adjust the figure size as per your image's aspect ratio
    plt.imshow(img)
    plt.axis('off')  # Turn off the axis

    # Set the title with the classification label
    plt.title(f"True Label: {true_label} | Predicted Label: {predicted_label_name}", color='black', fontsize=14)

    # Overlay the prediction percentages on the image
    prediction_text = '\n'.join([f'{label}: {prob:.2%}' for label, prob in zip(class_indices.keys(), probabilities)])
    plt.text(10, img.size[1] - 20, prediction_text, color='#39FF14', fontsize=12, weight='bold')  # Position text

    # Show the plot
    plt.show()