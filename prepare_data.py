### The dataset can be downloaded from here:
### https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


# Setting seeds for the generator
seed = 232
np.random.seed(seed)
tf.random.set_seed(seed)


input_path = 'C:/Users/marta/OneDrive/Desktop/Osnabruck/ImplementingANNswithTensorFlow/FinalProject/chest_xray/'


def prepare_data(img_dims, batch_size):

    # Augment data with a training data generation object and perform horizontal flipping as suggested by the paper
    train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, horizontal_flip=True)
    # Obtain the training data from a specific directory
    train_gen = train_datagen.flow_from_directory(
        directory=input_path+'train',
        batch_size=batch_size,
        target_size=(img_dims, img_dims),
        class_mode='binary',
        shuffle=True)

    # Augment data with a validation data generation object
    val_datagen = ImageDataGenerator(rescale=1./255)
    # Obtain the validation data from a specific directory
    val_gen = val_datagen.flow_from_directory(
        directory=input_path+'val', 
        target_size=(img_dims, img_dims), 
        batch_size=batch_size, 
        class_mode='binary', 
        shuffle=True)

    # Augment data with a testing data generation object
    test_datagen = ImageDataGenerator(rescale=1./255)
    # Obtain the testing data from a specific directory
    test_gen = test_datagen.flow_from_directory(
        directory=input_path+'test',
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)
    
    test_data = []
    test_labels = []

    # Preprocess images
    for cond in ['/NORMAL/', '/PNEUMONIA/']:
        for img in (os.listdir(input_path + 'test' + cond)):
            img = plt.imread(input_path+'test'+cond+img)
            img = cv2.resize(img, (img_dims, img_dims))
            img = np.dstack([img, img, img])
            img = img.astype('float32') / 255
            if cond=='/NORMAL/':
                label = 0
            elif cond=='/PNEUMONIA/':
                label = 1
            test_data.append(img)
            test_labels.append(label)
        
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    
    return train_gen, val_gen, test_gen, test_data, test_labels