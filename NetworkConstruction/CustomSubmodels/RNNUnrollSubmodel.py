import numpy as np
import tensorflow as tf
from ..CustomSubmodel import CustomSubmodel
from .FFSubmodel import FFSubmodel
from ..CustomLayers.DenseLayer import DenseLayer
from ..CustomLayers.GruLayer import GruLayer

class RNNUnrollSubmodel(CustomSubmodel):
    def __init__(self, model, **kwargs):
        super(RNNUnrollSubmodel, self).__init__(model, **kwargs)
        self.gru_hidden_s = kwargs['gru_hidden_sizes']
        self.std_hidden_s = kwargs['std_hidden_sizes']
        if not 'recurrent_type' in kwargs:
            r_func = GruLayer
        else:
            r_func = kwargs['recurrent_type']
        if not 'feedforward_type' in kwargs:
            f_func = tf.nn.relu
        else:
            f_func = kwargs['feedforward_type']
        self.recurrent_layers = [r_func(el, model=model) for el in self.gru_hidden_s]
        self.rnn_layers = tf.keras.layers.RNN(self.recurrent_layers)
        self.ff_layers = FFSubmodel(model, activ=f_func, hidden_sizes=self.std_hidden_s)
        self.output_layer = DenseLayer(kwargs['n_out'], model = model, func=tf.identity)

    def forward(self, x):
        out = self.rnn_layers(x)
        out = self.ff_layers.forward(out)
        out = self.output_layer(out)

        return out