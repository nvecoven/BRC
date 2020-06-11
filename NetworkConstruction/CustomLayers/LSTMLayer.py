import tensorflow as tf
from .. import CustomLayer

class LSTMLayer(CustomLayer.CustomLayer):
    def __init__(self, output_dim, model, var_list = None, **kwargs):
        self.output_size = output_dim
        self.lstm = tf.keras.layers.LSTMCell(output_dim)
        self.state_size = self.lstm.state_size
        super(LSTMLayer, self).__init__(output_dim, model, var_list, **kwargs)

    def build(self, input_shape):
        self.lstm.build(input_shape)
        super(LSTMLayer, self).build(input_shape)

    def call(self, input, states):
        return self.lstm(input, states)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        return self.lstm.get_initial_state(inputs=inputs, batch_size=batch_size, dtype=dtype)