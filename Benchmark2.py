# Minimal example for comparing nBRCs and BRCs to GRUs on benchmark 2. The network are built with
# the most simple Keras Sequential model. The only difference with standard RNNs lies in the recurrent cell definition.
# Note that the hyper-parameters used here might differ to those used in the paper, thus results might not be exactly
# the same. However this should not influence the conclusion and the observations in any way.

# Also note that on this benchmark, some networks need a few epochs before seeing a significant loss decrease.

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
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.state_size = output_dim
        super(NeuromodulatedBistableRecurrentCellLayer, self).__init__(output_dim, **kwargs)

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
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.state_size = output_dim
        super(BistableRecurrentCellLayer, self).__init__(output_dim, **kwargs)

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

###### DEFINE SAMPLES AS FOR THE DENOISING BENCHMARK #######
def generate_sample(n, data, no_end = 0):
    inp = []
    out = []
    choice = np.random.choice(n - no_end, data, replace = False)
    for i in range(n):
        d = np.random.randn()
        if i in choice:
            inp.append([d, 0])
            out.append(d)
        else:
            inp.append([d, -1])
    inp.append([0.0, 1])
    return inp, out

z = 400
no_ends = [0, 200]
n_data = 5

for no_end in no_ends:
    for t, c in zip(["GRU", "nBRC", "BRC"],
                [tf.keras.layers.GRUCell,
                 NeuromodulatedBistableRecurrentCellLayer,
                 BistableRecurrentCellLayer]):
        # Lower the number of test samples with respect to the paper, for faster evaluation.
        test_nbr = 5000
        model = None
        if not model is None:
            del (model)
        inputs = []
        outputs = []
        for i in range(50000):
            inp, out = generate_sample(z, n_data, no_end)
            inputs.append(inp)
            outputs.append(out)

        inputs = np.array(inputs)
        outputs = np.array(outputs)

        dataset = {'input': inputs.astype(np.float32)[:-test_nbr],
                   'output': outputs.astype(np.float32)[:-test_nbr]}

        test_dataset = {'input':inputs.astype(np.float32)[-test_nbr:],
                        'output':outputs.astype(np.float32)[-test_nbr:]}

        print("Training network with cells of type ", t, " with a lag of ", str(z), " time-steps and no-end set to ", str(no_end))
        print("---------------------")
        model = tf.keras.Sequential()
        recurrent_layers = [c(el) for el in [100, 100, 100, 100]]
        rnn = tf.keras.layers.RNN(recurrent_layers)
        model.add(rnn)
        model.add(tf.keras.layers.Dense(n_data))
        model.compile(optimizer="Adam", loss="mse")
        for e in range(80):
            model.fit(x=dataset['input'], y=dataset['output'], epochs = 1, batch_size=100,
                      validation_data=(test_dataset['input'], test_dataset['output']),
                      verbose = True)
            mse = model.evaluate(x=test_dataset['input'],y=test_dataset['output'])
            # Allows for faster run time. Comment for full training
            if mse < 0.1:
                break
