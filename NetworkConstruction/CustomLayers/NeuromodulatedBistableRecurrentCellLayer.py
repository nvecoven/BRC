import tensorflow as tf
from .. import CustomLayer

class NeuromodulatedBistableRecurrentCellLayer(CustomLayer.CustomLayer):
    def __init__(self, output_dim, model, var_list = None, **kwargs):
        self.output_dim = output_dim
        self.state_size = output_dim
        super(NeuromodulatedBistableRecurrentCellLayer, self).__init__(output_dim, model, var_list, **kwargs)

    def build(self, input_shape):
        self.kernelz = self.add_weight(name="kz", shape=(input_shape[1], self.output_dim), dtype=tf.float32,
                                      initializer='glorot_uniform')
        self.kernelr = self.add_weight(name="kr", shape=(input_shape[1], self.output_dim), dtype=tf.float32,
                                      initializer='glorot_uniform')
        self.kernelh = self.add_weight(name="kh", shape=(input_shape[1], self.output_dim), dtype=tf.float32,
                                      initializer='glorot_uniform')

        self.memoryz = self.add_weight(name="mz", shape=(self.output_dim, self.output_dim), dtype=tf.float32,
                                      initializer='orthogonal')
        self.memoryr = self.add_weight(name="mr", shape=(self.output_dim, self.output_dim), dtype=tf.float32,
                                      initializer='orthogonal')

        self.br = self.add_weight(name="br", shape=(self.output_dim,), dtype = tf.float32, initializer='zeros')
        self.bz = self.add_weight(name="bz", shape=(self.output_dim,), dtype = tf.float32, initializer='zeros')

        super(NeuromodulatedBistableRecurrentCellLayer, self).build(input_shape)

    def call(self, input, states):
        inp = input
        prev_out = states[0]
        z = tf.nn.sigmoid(tf.matmul(inp, self.kernelz) + tf.matmul(prev_out, self.memoryz) + self.bz)
        r = tf.nn.tanh(tf.matmul(inp, self.kernelr) + tf.matmul(prev_out, self.memoryr) + self.br)+1
        h = tf.nn.tanh(tf.matmul(inp, self.kernelh) + r * prev_out)
        output = (1.0 - z) * h + z * prev_out
        return output, [output]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        return [tf.zeros(shape=(batch_size, self.output_dim), dtype=dtype)]