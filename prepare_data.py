### The dataset can be downloaded from here:
### https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import random
import shutil


# Setting seeds for the generator
seed = 232
np.random.seed(seed)
tf.random.set_seed(seed)


input_path = 'C:/Users/marta/OneDrive/Desktop/Osnabruck/ImplementingANNswithTensorFlow/FinalProject/chest_xray/'

# Since the validation dataset size is so small, that it results in overfitting:
def split_data(validation_split):
    '''
    Splits the training data into training and validation sets for the Normal and Pneumonia classes.
    Moves shuffled 10% of the images from training set to validation set.

    Parameters:
    train_dir (str): path to the directory containing the original training data
    val_dir (str): path to the directory where the validation data will be saved
    validation_split (float): the percentage of images to move to the validation set

    '''
    # Set the directories for the normal and pneumonia classes
    normal_train_dir = input_path+'train/'+'NORMAL'
    pneumonia_train_dir = input_path+'train/'+'PNEUMONIA'
    normal_val_dir = input_path+'val/'+'NORMAL'
    pneumonia_val_dir = input_path+'val/'+'PNEUMONIA'

    # Get the list of filenames in the normal and pneumonia training directories
    normal_train_files = os.listdir(normal_train_dir)
    pneumonia_train_files = os.listdir(pneumonia_train_dir)

    # Shuffle the filenames randomly
    random.shuffle(normal_train_files)
    random.shuffle(pneumonia_train_files)

    # Calculate the number of images to move to the validation set
    num_normal_val = int(len(normal_train_files) * validation_split)
    num_pneumonia_val = int(len(pneumonia_train_files) * validation_split)

    # Move the selected images from the training directories to the validation directories
    if len(normal_train_files) == int(1341) and len(pneumonia_train_files) == 3875:
        for i in range(num_normal_val):
            filename = normal_train_files[i]
            src = os.path.join(normal_train_dir, filename)
            dst = os.path.join(normal_val_dir, filename)
            shutil.move(src, dst)

        for i in range(num_pneumonia_val):
            filename = pneumonia_train_files[i]
            src = os.path.join(pneumonia_train_dir, filename)
            dst = os.path.join(pneumonia_val_dir, filename)
            shutil.move(src, dst)
        print("Data split successful!")
    else:
        print("Data has already been split. Skipping data split.")


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
    
    return train_gen, val_gen, test_data, test_labels