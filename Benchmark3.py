# Minimal example for comparing nBRCs and BRCs to GRUs on benchmark 3. The network are built with
# the most simple Keras Sequential model. The only difference with standard RNNs lies in the recurrent cell definition.
# Note that the hyper-parameters used here might differ to those used in the paper for simplicity, thus results might not be exactly
# the same. However this should not influence the conclusion and the observations in any way.

# Note that for this benchmark, the architecture size has been reduced to allow for much faster training.
# Also, with this learning rate, there can be some numerical instabilities by unrolling such long time-series
# this is solved by ignoring too big gradients.

import tensorflow as tf
import numpy as np
import math

# Uncomment to run on CPU
tf.config.experimental.set_visible_devices([], 'GPU')
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

###

class MNISTModel(tf.keras.Sequential):
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        update = tf.constant(True)
        for g in gradients:
            if tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g)):
                update = tf.constant(False)
        if update:
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
###### DEFINE SAMPLES AS FOR THE MNIST BENCHMARK #######
(x_train_i, y_train_i), (x_test_i, y_test_i) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

x_train_i = np.reshape(x_train_i, [x_train_i.shape[0], -1, 1])
x_test_i = np.reshape(x_test_i, [x_test_i.shape[0], -1, 1])
y_train = tf.one_hot(y_train_i, 10)
y_test = tf.one_hot(y_test_i, 10)

zs = [0, 300]
for z in zs:
    x_train = np.concatenate([x_train_i, np.zeros((x_train_i.shape[0], z, x_train_i.shape[2]))], axis=1)
    x_test = np.concatenate([x_test_i, np.zeros((x_test_i.shape[0], z, x_test_i.shape[2]))], axis=1)
    for t, c in zip(["GRU", "nBRC", "BRC"],
                    [tf.keras.layers.GRUCell,
                     NeuromodulatedBistableRecurrentCellLayer,
                     BistableRecurrentCellLayer]):
        model = None
        if not model is None:
            del (model)

        dataset = {'input': tf.cast(x_train, tf.float32) / 255., 'groundtruth': tf.cast(y_train, tf.float32)}
        test_dataset = {'input': tf.cast(x_test, tf.float32) / 255., 'groundtruth': tf.cast(y_test, tf.float32)}

        print("Training network with cells of type ", t, " with ", str(z), " black pixels appended")
        print("---------------------")
        model = MNISTModel()
        recurrent_layers = [c(el) for el in [300, 300]]
        rnn = tf.keras.layers.RNN(recurrent_layers)
        model.add(rnn)
        model.add(tf.keras.layers.Dense(10, activation="softmax"))
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
                      metrics=['accuracy'])
        
        for e in range(20):
            model.fit(x=dataset['input'], y=dataset['groundtruth'], epochs = e+1, batch_size=300,
                      validation_data=(test_dataset['input'], test_dataset['groundtruth']),
                      verbose = True, initial_epoch = e)
            accuracy = model.evaluate(x=test_dataset['input'],y=test_dataset['groundtruth'])[1]
            # Allows for faster runtimes, comment for full training.
            if accuracy > 0.7:
                break
