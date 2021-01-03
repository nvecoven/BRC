import tensorflow as tf
import pickle

class CustomModel(tf.keras.Model):
    # Need this to handle ragged tensor .... not yet supported by keras default losses.
    def set_loss(self, loss):
        self.custom_loss = loss

    def set_ypred_processing(self, process_ypred):
        self.process_ypred = process_ypred

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            y, y_pred = self.process_ypred(y, y_pred)
            loss = self.custom_loss(y, y_pred)
            loss += tf.reduce_sum(self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        update = tf.constant(True)
        for g in gradients:
            if tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g)):
                update = tf.constant(False)
        if update:
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        for m in self.metrics:
            m.reset_states()
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        y, y_pred = self.process_ypred(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

class AccuracyMetric(tf.keras.metrics.Metric):
    def __init__(self, name = "Accuracy_metric", **kwargs):
        super(AccuracyMetric, self).__init__(name = name, **kwargs)
        self.accuracy = self.add_weight(name="accuracy", initializer="zeros")
        self.cnt = self.add_weight(name="cnt", initializer="zeros")

    def update_state(self, y_true, y_pred, *args, **kwargs):
        y_true = tf.ragged.map_flat_values(lambda x: tf.argmax(x, axis=-1), y_true)
        y_pred = tf.ragged.map_flat_values(lambda x : tf.argmax(x, axis = -1), y_pred)
        comp = tf.cast(tf.equal(y_true, y_pred), tf.float32)
        self.accuracy.assign_add(tf.reduce_mean(comp))
        self.cnt.assign(self.cnt + 1)

    def reset_states(self):
        self.accuracy.assign(0)
        self.cnt.assign(0)

    def result(self):
        to_return = self.accuracy / self.cnt
        return to_return

class CustomRNN():
    def __init__(self, dataset, rnn_layers_sizes, decoder_layer_sizes, cell_type, save_directory):
        self.dataset = dataset
        input = tf.keras.layers.Input(shape=[None, dataset.input_size], dtype=tf.float32, ragged=True)
        if dataset.output_is_sequence:
            # Needed if groundtruth is only the last X elements of the time-serie
            def process_ypred(y_true, y_pred):
                gt_rs = y_true.row_splits
                pred_rs = y_pred.row_splits
                nbr_gt = tf.roll(gt_rs, -1, axis = 0) - gt_rs
                gather_ind = tf.ragged.range(pred_rs[1:]-nbr_gt[:-1], pred_rs[1:]).flat_values
                y_pred = tf.RaggedTensor.from_row_splits(tf.gather(y_pred.flat_values, gather_ind, axis = 0),
                                                         row_splits=gt_rs)
                return y_true, y_pred

            def create_layer(layer):
                return tf.keras.layers.TimeDistributed(layer)
        else:
            def process_ypred(y_true, y_pred):
                return y_true, y_pred

            def create_layer(layer):
                return layer

        x = input
        r_cells_list = [cell_type(size) for size in rnn_layers_sizes]
        x = tf.keras.layers.RNN(r_cells_list, return_sequences=dataset.output_is_sequence)(x)
        # There for analysis and graphs making
        self.recurrent_layers = tf.keras.layers.RNN(r_cells_list, return_sequences=True, return_state=True)

        for size in decoder_layer_sizes:
            x = create_layer(tf.keras.layers.Dense(size, activation=tf.nn.relu))(x)
        if self.dataset.is_regression:
            x = create_layer(tf.keras.layers.Dense(self.dataset.output_size))(x)
            metrics = ["MSE"]
            if dataset.output_is_sequence:
                def loss(y_true, y_pred):
                    return tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.square(y_pred - y_true), [1, 2])))
            else:
                def loss(y_true, y_pred):
                    return tf.reduce_mean(tf.square(y_pred - y_true))
        else:
            x = create_layer(tf.keras.layers.Dense(self.dataset.output_size, activation="softmax"))(x)
            if dataset.output_is_sequence:
                def loss(y_true, y_pred):
                    epsilon_ = 1e-6
                    output = tf.clip_by_value(y_pred, epsilon_, 1. - epsilon_)
                    return tf.reduce_mean(tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(output), 2),[-1]))
            else:
                def loss(y_true, y_pred):
                    epsilon_ = 1e-6
                    output = tf.clip_by_value(y_pred, epsilon_, 1. - epsilon_)
                    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(output), 1))

            metrics = [AccuracyMetric()]

        class LossMetric(tf.keras.metrics.Metric):
            def __init__(self, name="Loss_metric", **kwargs):
                super(LossMetric, self).__init__(name=name, **kwargs)
                self.loss = self.add_weight(name="loss", initializer="zeros")
                self.cnt = self.add_weight(name="cnt", initializer="zeros")

            def update_state(self, y_true, y_pred, *args, **kwargs):
                self.loss.assign_add(loss(y_true, y_pred))
                self.cnt.assign(self.cnt + 1)

            def reset_states(self):
                self.loss.assign(0)
                self.cnt.assign(0)

            def result(self):
                to_return = self.loss / self.cnt
                return to_return

        metrics += [LossMetric()]
        self.model = CustomModel(inputs=input, outputs=x)
        self.model.set_loss(loss)
        self.model.set_ypred_processing(process_ypred)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), metrics=metrics)

        class HistoryCallback(tf.keras.callbacks.Callback):
            def __init__(self, path, **kwargs):
                self.path = path
                super(HistoryCallback, self).__init__(**kwargs)

            def on_epoch_end(self, epoch, logs=None):
                pickle.dump(self.model.history.history, open(self.path, "wb"))

        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=save_directory + '/' + cell_type.__name__ + '/model.h5', save_weights_only=True),
            tf.keras.callbacks.TensorBoard(log_dir=save_directory + '/' + cell_type.__name__ +'/logs/', update_freq="batch"),
            tf.keras.callbacks.CSVLogger(save_directory + '/' + cell_type.__name__  + "/logs/train_logs.csv", separator=",", append=False),
            HistoryCallback(save_directory + '/' + cell_type.__name__ + "/history")
        ]

    def train(self, nbr_epcohs):
        self.model.fit(self.dataset.train_set, epochs=nbr_epcohs,
                       validation_data=self.dataset.test_set, callbacks=self.callbacks)


