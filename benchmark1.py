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

def generate_sample(n):
    true_n = np.random.randn()
    chain = np.concatenate([[true_n], np.random.randn(n-1)])
    return chain, true_n

zs = [5, 300, 600]

for z in zs:
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
            inp, out = generate_sample(z)
            inputs.append(inp)
            outputs.append(out)

        inputs = np.array(inputs)
        outputs = np.array(outputs)

        dataset = {'input': np.expand_dims(inputs, axis=2).astype(np.float32)[:-test_nbr],
                   'output': np.expand_dims(outputs, axis=1).astype(np.float32)[:-test_nbr]}

        test_dataset = {'input':np.expand_dims(inputs, axis = 2).astype(np.float32)[-test_nbr:],
                        'output':np.expand_dims(outputs, axis = 1).astype(np.float32)[-test_nbr:]}


        model = RNN(params={'output_dim': 1, 'lr': 1e-3, 'hidden_sizes': [100, 100],
                            'recurrent_type':c}, load=False,
                    save_path="./benchmark1_results/", name='ANN_type_' + t + "_z_" + str(z))

        model.train(dataset, steps=30000, display_init = 10, func=model.mse, batch_size=100, compiled=False,
                    val_data=test_dataset, checkpoint_every=100, call_info=100)
