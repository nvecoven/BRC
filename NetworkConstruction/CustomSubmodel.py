import tensorflow as tf

class CustomSubmodel(tf.keras.Model):
    def __init__(self, model, vars_list = None, **kwargs):
        super(CustomSubmodel, self).__init__(**{})
        self.model = model
