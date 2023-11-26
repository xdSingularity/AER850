# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 02:51:43 2023

@author: ilanb
"""

#if the code has to be run elsewhere, all location file paths must be changed to the appropriate ones
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Paths to the dataset
folder_path = 'C:\\Users\\ilanb\\OneDrive\\Documents\\GitHub\\AER850\\Project 2 Data\\Data'  # Update this path to the location of your dataset
train_dir = os.path.join(folder_path, 'Train')
validation_dir = os.path.join(folder_path, 'Validation')

# Data augmentation for the training images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Validation data should not be augmented!
validation_datagen = ImageDataGenerator(rescale=1./255)

# Prepare data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

# Calculate class weights which are used to handle imbalanced datasets
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))

# Build the model
model = Sequential([
    Conv2D(16, (3, 3), input_shape=(100, 100, 3), activation='relu'),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.65),
    Dense(4, activation='softmax')
])

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Set up early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    train_generator,
    epochs=40,  # Adjust the number of epochs as needed
    validation_data=validation_generator,
    callbacks=[early_stopping],
    class_weight=class_weights_dict
)

# Save the trained model
model.save('model.h5')

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')

# Plotting training and validation loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
class_indices = train_generator.class_indices
with open('C:\\Users\\ilanb\\OneDrive\\Documents\\GitHub\\AER850\\\class_indices.json', 'w') as class_file:
    json.dump(class_indices, class_file)
