import tensorflow as tf
from .. import CustomLayer

class GruLayer(CustomLayer.CustomLayer):
    def __init__(self, output_dim, model, var_list = None, **kwargs):
        self.output_size = output_dim
        self.state_size = output_dim
        self.gru = tf.keras.layers.GRUCell(output_dim, reset_after=True)
        super(GruLayer, self).__init__(output_dim, model, var_list, **kwargs)

    def build(self, input_shape):
        self.gru.build(input_shape)
        super(GruLayer, self).build(input_shape)

    def call(self, input, states):
        return self.gru(input, states)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        return self.gru.get_initial_state(inputs=inputs, batch_size=batch_size, dtype=dtype)