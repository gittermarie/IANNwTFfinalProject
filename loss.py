import tensorflow as tf

class WeightedCrossEntropyBinaryLoss:
  """
  Class for weighted cross-entropy binary loss.
  """
  def __init__(self, zero_weight, one_weight):
    self.zero_weight = zero_weight
    self.one_weight = one_weight
  
  def weighted_binary_crossentropy(self, y, yhat):

    # Compute the weighted crossentropy binary loss
    fuzzy_factor = tf.convert_to_tensor(tf.keras.backend.epsilon(), dtype=yhat.dtype.base_dtype) # fuzzy factor to make calculations more accurate
    yhat = tf.clip_by_value(yhat, fuzzy_factor, 1 - fuzzy_factor)
    weighted_bce = -self.one_weight * y * tf.math.log(yhat) \
        -self.zero_weight * (1 - y) * tf.math.log(1 - yhat)

    # Return the mean error
    return tf.reduce_mean(weighted_bce)
  
