import numpy as np
import tensorflow as tf
from ..CustomSubmodel import CustomSubmodel
from ..CustomLayers.DenseLayer import DenseLayer
from ..CustomLayers.GruLayer import GruLayer

class FFSubmodel(CustomSubmodel):
    def __init__(self, model, **kwargs):
        super(FFSubmodel, self).__init__(model, **kwargs)
        self.std_hidden_s = kwargs['hidden_sizes']
        if 'activ' in kwargs:
            activ = kwargs['activ']
        else:
            activ = tf.nn.relu
        self.std_layers = []
        for hidden in self.std_hidden_s:
            self.std_layers.append(DenseLayer(hidden, model=model, func=activ))

    def forward(self, x):
        out = x
        for stdl in self.std_layers:
            out = stdl(out)

        return out