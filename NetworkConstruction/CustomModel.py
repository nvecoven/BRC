import tensorflow as tf
import numpy as np
import datetime
import json
import pickle
import os
import time

class CustomModel():
    def __init__(self, save_path, load = False, name = None, params = None, not_sequential_data = None, **kwargs):
        self.variables_d = {}
        self.variables = []
        self.queue_load_weights = {}
        self.whole_variables_d = {}

        self.kwargs = kwargs
        layers = self.define_layers(params = params, **kwargs)
        self.create_optimizer(params)
        layers['opt'] = self.opt

        self.checkpoint = tf.train.Checkpoint(**layers)

        self.to_pickle = {}
        if not load:
            self.to_pickle['name'] = name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            self.to_pickle['name'] = name

        if not load:
            self.to_pickle['batch_nbr'] = 1
            self.to_pickle['save_path'] = save_path
        else:
            self.restore_model(save_path)

        self.create_writers()

        self.losses_d = {}
        self.outputs_d = {}

    def define_layers(self, params, **kwargs):
        raise NotImplementedError("Please Implement this method")

    def create_writers(self):
        self.train_sum = tf.summary.create_file_writer(self.to_pickle['save_path'] + self.to_pickle['name'] +
                                                       "/log/train")
        self.test_sum = tf.summary.create_file_writer(self.to_pickle['save_path'] + self.to_pickle['name'] +
                                                      "/log/test")
        self.inf_sum = tf.summary.create_file_writer(self.to_pickle['save_path'] + self.to_pickle['name'] +
                                                      "/log/infos")

    def create_optimizer(self, params):
        self.opt = tf.optimizers.Adam(learning_rate = 1e-4)

    def store_model(self):
        dir = self.to_pickle['save_path'] + self.to_pickle['name']
        if not os.path.exists(dir):
            os.makedirs(dir)

        self.checkpoint.save(dir+"/params/")
        pickle.dump(self.to_pickle, open(dir + "/infos", "wb"))

    def restore_model(self, path):
        self.checkpoint.restore(tf.train.latest_checkpoint(path + self.to_pickle['name'] + "/params/"))
        self.to_pickle = pickle.load(open(path + self.to_pickle['name'] + "/infos", "rb"))

    def get_gradients(self, samples, func, variables_string = None, variables_func =  None, compiled = True,
                      grad_post_process_func = None, require_init = False, **kwargs):
        if grad_post_process_func is None:
            grad_post_process_func = self.post_process_gradients
        if require_init:
            outputs = func(samples)

        to_train_vars = self.variables
        if not (variables_string is None):
            to_train_vars = sum([self.variables_d[el] for el in variables_string], [])
        if not (variables_func is None):
            to_train_vars = variables_func()

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(to_train_vars)
            outputs = func(samples)

            output_sum = tf.add_n([outputs[el] for el in outputs])
            loss = tf.reduce_mean(output_sum)

        current_gradient = tape.gradient(loss, to_train_vars)
        with self.train_sum.as_default():
            tf.summary.scalar(func.__name__ + "/_total", loss, step=self.to_pickle['batch_nbr'])

        self.to_pickle['batch_nbr'] += 1
        return grad_post_process_func(current_gradient)

    def post_process_gradients(self, grads_list):
        return grads_list

    def process_sample(self, sample):
        return sample

    def evaluate(self, input_data, func, compiled = True, batch_size = 50, complete_unroll = True, process_function = None,
                 **kwargs):
        if process_function is None:
            process_function = self.process_sample
        dataset = tf.data.Dataset.from_tensor_slices(input_data).batch(batch_size).map(process_function).prefetch(tf.data.experimental.AUTOTUNE)
        first = True
        loss = {}
        c = 0
        for cnt, samples in enumerate(dataset):
            outputs = func(samples)
            cur_losses = {}
            for el in outputs:
                cur_losses[el] = tf.reduce_mean(outputs[el])
            cur_losses['_total'] = tf.add_n([cur_losses[el] for el in cur_losses])
            if first:
                loss = {el:cur_losses[el] for el in cur_losses}
                first = False
            else:
                for el in loss:
                    loss[el] += cur_losses[el]
            c += 1
        for el in loss:
            loss[el] = loss[el]/c
        with self.test_sum.as_default():
            for el in loss:
                tf.summary.scalar(func.__name__ + "/" + el, loss[el], step = self.to_pickle['batch_nbr'])
        return loss

    def train(self, input_data, func, steps = 100, batch_size = 50, max_shuffle_buffer = 1000,
              variables_string = None, variables_func = None, epoch_mode = False, val_data = None, compiled = True, checkpoint_every = 1000,
              display_init = 100, complete_unroll = True, process_function = None, grad_post_process_func = None, call_info = None, info_func = None, val_func = None,
              **kwargs):
        if call_info is None:
            call_info = checkpoint_every
        if val_func is None:
            val_func = func
        if process_function is None:
            process_function = self.process_sample
        dataset = tf.data.Dataset.from_tensor_slices(input_data)
        dataset = dataset.shuffle(max_shuffle_buffer).batch(batch_size, drop_remainder=True).map(process_function)
        dataset = dataset.repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        if epoch_mode:
            display = 1
        else:
            display = display_init
        old = time.time()

        for cnt, samples in enumerate(dataset):
            if epoch_mode:
                cnt_c = (cnt * batch_size) // input_data.shape[0]
            else:
                cnt_c = cnt

            if cnt_c > steps:
                break
            if cnt_c % display == 0 and not cnt_c == 0:
                new = time.time()
                print("Training iteration .............", cnt_c, " ..... average iteration time = ", (new-old)/display)
                old = new
                if not val_data is None:
                    test_loss = self.evaluate(val_data, val_func, compiled, batch_size, complete_unroll=complete_unroll,
                                              process_function=process_function)
                    for k, v in test_loss.items():
                        print("\t Testing loss {}: {:4f}".format(k, v.numpy()))

                if cnt_c == display * 10:
                    display *= 10
            # Make sure variables are initialized
            if cnt == 0:
                grads = self.get_gradients(samples, func, variables_string, variables_func,
                                           compiled=compiled, complete_unroll=complete_unroll,
                                           grad_post_process_func=grad_post_process_func, require_init=True, **kwargs)
            grads = self.get_gradients(samples, func, variables_string, variables_func,
                                       compiled = compiled, complete_unroll=complete_unroll,
                                       grad_post_process_func = grad_post_process_func, **kwargs)
            to_train_vars = self.variables
            if not (variables_string is None):
                to_train_vars = sum([self.variables_d[el] for el in variables_string], [])
            if not (variables_func is None):
                to_train_vars = variables_func()
            if cnt % call_info == 0 and info_func is not None:
                info_func(cnt)
            self.opt.apply_gradients(zip(grads, to_train_vars))

            if (cnt+1) % checkpoint_every == 0:
                self.store_model()

    def __call__(self, input_data, func, batch_size = 50, compiled = True, complete_unroll = True, process_function = None, **kwargs):
        if process_function is None:
            process_function = self.process_sample
        dataset = tf.data.Dataset.from_tensor_slices(input_data)
        dataset = dataset.batch(batch_size, drop_remainder=False).map(process_function).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        first = True
        whole_outputs = {}
        for samples in dataset:
            outputs = func(samples)
            if first:
                whole_outputs = {el:[outputs[el]] for el in outputs}
                first = False
            else:
                for el in whole_outputs:
                    whole_outputs[el].append(outputs[el])

        whole_outputs = {el:tf.concat(whole_outputs[el], axis = 0) for el in whole_outputs}
        return whole_outputs