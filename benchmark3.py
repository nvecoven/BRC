from NetworkConstruction.CustomSubmodels.RNNUnrollSubmodel import RNNUnrollSubmodel
from NetworkConstruction.CustomModel import CustomModel
from NetworkConstruction.CustomLayers import GruLayer, NeuromodulatedBistableRecurrentCellLayer, LSTMLayer, BistableRecurrentCellLayer
import tensorflow as tf
import numpy as np

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

class RNN(CustomModel):
    def define_layers(self, params):
        self.output_dim = params['output_dim']
        self.rnn = RNNUnrollSubmodel(model=self, gru_hidden_sizes=params['hidden_sizes'],
                                     std_hidden_sizes=[], n_out = params['output_dim'],
                                     recurrent_type=params['recurrent_type'])
        std_layers = {'bistablernn': self.rnn}
        return {**std_layers}

    @tf.function
    def cross_entropy(self, samples, **kwargs):
        observations = samples['groundtruth']
        out = self.rnn.forward(samples['input'])
        return {'cross-entropy': tf.nn.softmax_cross_entropy_with_logits(labels=observations, logits=out)}

    @tf.function
    def accuracy(self, samples):
        observations = samples['groundtruth']
        out = self.rnn.forward(samples['input'])
        return {'accuracy': tf.metrics.categorical_accuracy(y_true=observations, y_pred=out)}

    def pred(self, samples):
        out = self.rnn.forward(samples['input'])
        return {'pred':out}

    def create_optimizer(self, params):
        self.opt = tf.optimizers.Adam(learning_rate=params['lr'])


(x_train_i, y_train_i), (x_test_i, y_test_i) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

x_train_i = np.reshape(x_train_i, [x_train_i.shape[0], -1, 1])
x_test_i = np.reshape(x_test_i, [x_test_i.shape[0], -1, 1])
y_train = tf.one_hot(y_train_i, 10)
y_test = tf.one_hot(y_test_i, 10)

zs = [0, 300]
for z in zs:
    x_train = np.concatenate([x_train_i, np.zeros((x_train_i.shape[0], z, x_train_i.shape[2]))], axis=1)
    x_test = np.concatenate([x_test_i, np.zeros((x_test_i.shape[0], z, x_test_i.shape[2]))], axis=1)
    for t, c in zip(["nBRC", "BRC", "GRU", "LSTM"],
                    [NeuromodulatedBistableRecurrentCellLayer.NeuromodulatedBistableRecurrentCellLayer,
                     BistableRecurrentCellLayer.BistableRecurrentCellLayer,
                     GruLayer.GruLayer,
                     LSTMLayer.LSTMLayer]):
        model = None
        if not model is None:
            del (model)

        dataset = {'input': tf.cast(x_train, tf.float32) / 255., 'groundtruth': tf.cast(y_train, tf.float32)}
        test_dataset = {'input': tf.cast(x_test, tf.float32) / 255., 'groundtruth': tf.cast(y_test, tf.float32)}
        val_dataset = {'input': tf.cast(x_test[:50, :, :], tf.float32) / 255.,
                       'groundtruth': tf.cast(y_test[:50, :], tf.float32)}

        model = RNN(params={'output_dim': 10, 'lr': 1e-3, 'hidden_sizes': [100, 100, 100, 100],
                            'recurrent_type':c}, load=False,
                    save_path="./benchmark3/", name='ANN_type_' + t + "_blackpixels_" + str(z))

        model.train(dataset, steps=40000, display_init = 100, func=model.cross_entropy, batch_size=100, compiled=False,
                    val_data=test_dataset, checkpoint_every=100, val_func=model.accuracy)
