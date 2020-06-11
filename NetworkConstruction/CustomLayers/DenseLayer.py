import tensorflow as tf
from .. import CustomLayer

class DenseLayer(CustomLayer.CustomLayer):
    def __init__(self, output_dim, model, var_list = None, func = tf.identity, **kwargs):
        self.output_dim = output_dim
        self.l = tf.keras.layers.Dense(output_dim, activation=func);
        super(DenseLayer, self).__init__(output_dim, model, var_list, **kwargs)

    def build(self, input_shape):
        self.l.build(input_shape)
        super(DenseLayer, self).build(input_shape)

    def call(self, x):
        return self.l(x)

    def zero_state(self, batch_size):
        return tf.zeros(shape = (batch_size, self.output_dim), dtype = tf.float32)