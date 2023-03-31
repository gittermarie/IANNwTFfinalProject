import tensorflow as tf
import numpy as np
from resNet import ResNet
from evaluate import evaluate
import prepare_data
import os

N_EPOCHS = 10
BATCH_SIZE = 16
IMG_DIMS = 224
CHECKPOINT_FILEPATH = './checkpoint/weights_resnet.hdf5'
log_dir='./logs'
train_data_path = "./archive/chest_xray/train"
val_data_path = "./archive/chest_xray/val"
test_data_path= "./archive/chest_xray/test"

# Instantiate the ResNet model
resnet = ResNet()

# Adding files from the training dataset since the current validation set is only 0.03% of the training set
prepare_data.split_data(0.1, train_data_path, val_data_path)

# Getting the data
train_gen, val_gen, test_data, test_labels = prepare_data.prepare_data(IMG_DIMS, BATCH_SIZE, train_data_path, val_data_path, test_data_path)

# Compute class distribution of pneumonia vs normal images
resnet.get_weights(train_data_path)

# Create and compile the DenseNet121 model
model = resnet.get_model()

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                                                 patience=1, verbose=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.1, 
                                              patience=1, mode='min')
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_FILEPATH, 
                                                      save_weights_only=False,
                                                      mode='min', save_best_only=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch')

history = model.fit(train_gen, epochs=N_EPOCHS, batch_size=BATCH_SIZE, 
                    validation_data=val_gen, 
                    callbacks=[reduce_lr, model_checkpoint, tensorboard_callback])

# Load the model with the lowest validation loss
model.load_weights(CHECKPOINT_FILEPATH)

# evaluate performance
evaluate(model, test_data, test_labels, BATCH_SIZE)