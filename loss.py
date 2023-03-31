import tensorflow as tf
from keras.backend import epsilon
import tensorflow.python.keras.backend as K


class WeightedCrossEntropyBinaryLoss:
    """
    Class for weighted cross-entropy binary loss.
    """
    def __init__(self, zero_weight, one_weight):
        self.zero_weight = zero_weight
        self.one_weight = one_weight

    def weighted_binary_crossentropy(self, y_true, y_pred):
        y_true = K.cast(y_true, dtype=tf.float32)

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Compute cross entropy from probabilities.
        bce = y_true * tf.math.log(y_pred + epsilon)
        bce += (1 - y_true) * tf.math.log(1 - y_pred + epsilon)
        bce = -bce

        # Apply the weights to each class individually
        weight_vector = y_true * self.one_weight + (1. - y_true) * self.zero_weight
        weighted_bce = weight_vector * bce

        # Return the mean error
        return tf.reduce_mean(weighted_bce)

