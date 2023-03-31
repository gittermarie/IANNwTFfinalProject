import tensorflow as tf
import numpy as np
from loss import WeightedCrossEntropyBinaryLoss
from keras.layers import Input, Dense
from keras.models import Model
import os
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

IMG_DIMS = 224
BATCH_SIZE = 16 
OUTPUT_SIZE = 1

class ResNet(tf.keras.Model):
    """
    The ResNet model that uses keras ResNet101 as its backbone.
    """

    def __init__ (self):
        """
        The constructor instantiates the weights and the model.
        """
        super().__init__()

        # Instantiate weights and steps and set them to None
        self.zero_weight = None
        self.one_weight = None
        self.train_steps = None
        self.val_steps = None

        # get_model() will initialize this to ResNet101 model
        self.model = None

    def get_weights(self, train_data_path):
        """
        Computes class distribution of pneumonia vs normal images.

        Args:
          train_data_path: path to training data.
          val_data_path: path to validation data.
        """

        # Count images in each class in the train data
        n_normal = len(os.listdir(train_data_path + '/NORMAL'))
        n_pneumonia = len(os.listdir(train_data_path + '/PNEUMONIA'))

        # Compute class distribution
        self.one_weight = float(n_normal)/(n_normal+n_pneumonia)
        self.zero_weight = float(n_pneumonia)/(n_normal+n_pneumonia)


    def get_model(self):

        # ResNet101 expects number of channels to be 3
        input = Input(shape=(IMG_DIMS, IMG_DIMS, 3), batch_size=BATCH_SIZE)

        # using pretrained ResNet101 as the foundation of the model
        base_model = tf.keras.applications.ResNet101(include_top=False, weights='imagenet',
                                                     input_shape=(IMG_DIMS, IMG_DIMS, 3), 
                                                     pooling='avg')

        # Add custom output layers
        x = base_model.output
        x = tf.keras.layers.Dense(OUTPUT_SIZE, activation='sigmoid')(x)

        self.model = tf.keras.models.Model(inputs=base_model.input, outputs=x)

        # Use weighted binary crossentropy loss
        loss = WeightedCrossEntropyBinaryLoss(self.zero_weight, self.one_weight)

        # Compile the model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999),
                           loss=loss.weighted_binary_crossentropy,
                           metrics=['accuracy'])

        return self.model
    