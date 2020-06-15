import sys
from NetworkConstruction.CustomSubmodels.RNNUnrollSubmodel import RNNUnrollSubmodel
from NetworkConstruction.CustomModel import CustomModel
from NetworkConstruction.CustomLayers import GruLayer, BistableRecurrentCellLayer, NeuromodulatedBistableRecurrentCellLayer, LSTMLayer, BistableRecurrentCellLayer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle

tf.config.experimental.set_visible_devices([], 'GPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:          # Currently, memory growth needs to be the same across GPUs
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
        std_layers = {'rnn': self.rnn}
        return {**std_layers}

    def mse(self, samples, **kwargs):
        observations = samples['output']
        out = self.rnn.forward(samples['input'])
        return {'MSE':tf.square(observations-out)}

    def pred(self, samples):
        out = self.rnn.forward(samples['input'])
        return {'pred':out}

    def create_optimizer(self, params):
        self.opt = tf.optimizers.Adam(learning_rate=params['lr'])

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
    for t, c in zip(["nBRC", "BRC", "GRU", "LSTM"],
                [NeuromodulatedBistableRecurrentCellLayer.NeuromodulatedBistableRecurrentCellLayer,
                 BistableRecurrentCellLayer.BistableRecurrentCellLayer,
                 GruLayer.GruLayer,
                 LSTMLayer.LSTMLayer]):
        test_nbr = 50000
        model = None
        if not model is None:
            del (model)
        inputs = []
        outputs = []
        for i in range(95000):
            inp, out = generate_sample(z, n_data, no_end)
            inputs.append(inp)
            outputs.append(out)

        inputs = np.array(inputs)
        outputs = np.array(outputs)

        dataset = {'input': inputs.astype(np.float32)[:-test_nbr],
                   'output': outputs.astype(np.float32)[:-test_nbr]}

        test_dataset = {'input':inputs.astype(np.float32)[-test_nbr:],
                        'output':outputs.astype(np.float32)[-test_nbr:]}


        model = RNN(params={'output_dim': n_data, 'lr': 1e-3, 'hidden_sizes': [100, 100, 100, 100],
                            'recurrent_type':c}, load=False,
                    save_path="./benchmark2_results/", name='ANN_type_' + t + "_noend_" + str(no_end))

        model.train(dataset, steps=40000, display_init = 10, func=model.mse, batch_size=100, compiled=False,
                    val_data=test_dataset, checkpoint_every=100, call_info=100)