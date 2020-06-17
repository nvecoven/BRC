# Minimal example for comparing nBRCs and BRCs to GRUs on benchmark requiring long memory. The network are built with
# the most simple Keras Sequential model. The only difference lies in the recurrent cell definition.
# Note that the hyper-parameters used here might differ to those used in the paper, thus results might not be exactly
# the same. However this does not influence the conclusion and the observations in any way.

import tensorflow as tf
import numpy as np

# Uncomment to run on CPU
# tf.config.experimental.set_visible_devices([], 'GPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

##### DEFINE THE NBRC ###########
class NeuromodulatedBistableRecurrentCellLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, var_list = None, **kwargs):
        self.output_dim = output_dim
        self.state_size = output_dim
        super(NeuromodulatedBistableRecurrentCellLayer, self).__init__(output_dim, var_list, **kwargs)

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

############ DEFINE THE BRC #################
class BistableRecurrentCellLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, var_list = None, **kwargs):
        self.output_dim = output_dim
        self.state_size = output_dim
        super(BistableRecurrentCellLayer, self).__init__(output_dim, var_list, **kwargs)

    def build(self, input_shape):
        self.kernelz = self.add_weight(name="kz", shape=(input_shape[1], self.output_dim), dtype=tf.float32,
                                      initializer='glorot_uniform')
        self.kernelr = self.add_weight(name="kr", shape=(input_shape[1], self.output_dim), dtype=tf.float32,
                                      initializer='glorot_uniform')
        self.kernelh = self.add_weight(name="kh", shape=(input_shape[1], self.output_dim), dtype=tf.float32,
                                      initializer='glorot_uniform')

        self.memoryz = self.add_weight(name="mz", shape=(self.output_dim,), dtype=tf.float32,initializer=tf.keras.initializers.constant(1.0))
        self.memoryr = self.add_weight(name="mr", shape=(self.output_dim,), dtype=tf.float32,initializer=tf.keras.initializers.constant(1.0))

        self.br = self.add_weight(name="br", shape=(self.output_dim,), dtype = tf.float32, initializer='zeros')
        self.bz = self.add_weight(name="bz", shape=(self.output_dim,), dtype = tf.float32, initializer='zeros')

        super(BistableRecurrentCellLayer, self).build(input_shape)

    def call(self, input, states):
        inp = input
        prev_out = states[0]
        r = tf.nn.tanh(tf.matmul(inp, self.kernelr) + prev_out * self.memoryr + self.br) + 1
        z = tf.nn.sigmoid(tf.matmul(inp, self.kernelz) + prev_out * self.memoryz + self.bz)
        output = z * prev_out + (1.0 - z) * tf.nn.tanh(tf.matmul(inp, self.kernelh) + r * prev_out)
        return output, [output]


    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        return [tf.zeros(shape=(batch_size, self.output_dim), dtype=dtype)]

###### DEFINE SAMPLES AS FOR THE COPY INPUT BENCHMARK #######
def generate_sample(n):
    true_n = np.random.randn()
    chain = np.concatenate([[true_n], np.random.randn(n-1)])
    return chain, true_n

zs = [5, 300, 600]
for z in zs:
    for t, c in zip(["GRU", "nBRC", "BRC"],
                [tf.keras.layers.GRUCell,
                 NeuromodulatedBistableRecurrentCellLayer,
                 BistableRecurrentCellLayer]):
        # Lower the number of test samples with respect to the paper, for faster training.
        test_nbr = 5000
        model = None
        if not model is None:
            del (model)
        inputs = []
        outputs = []
        for i in range(50000):
            inp, out = generate_sample(z)
            inputs.append(inp)
            outputs.append(out)

        inputs = np.array(inputs)
        outputs = np.array(outputs)

        dataset = {'input': np.expand_dims(inputs, axis=2).astype(np.float32)[:-test_nbr],
                   'output': np.expand_dims(outputs, axis=1).astype(np.float32)[:-test_nbr]}

        test_dataset = {'input':np.expand_dims(inputs, axis = 2).astype(np.float32)[-test_nbr:],
                        'output':np.expand_dims(outputs, axis = 1).astype(np.float32)[-test_nbr:]}

        print("Training network with cells of type ", t, " with a lag of ", str(z), " time-steps")
        print("---------------------")
        model = tf.keras.Sequential()
        recurrent_layers = [c(el) for el in [100, 100]]
        rnn = tf.keras.layers.RNN(recurrent_layers)
        model.add(rnn)
        model.add(tf.keras.layers.Dense(1))
        model.compile(optimizer="Adam", loss="mse")
        model.fit(x=dataset['input'], y=dataset['output'], epochs = 60, batch_size=100,
                  validation_data=(test_dataset['input'], test_dataset['output']),
                  verbose = True)
