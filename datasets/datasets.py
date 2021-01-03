import tensorflow as tf
import numpy as np
import gmpy

class Dataset:
    def __init__(self, nbr_train, nbr_test, batch_size):
        self.nbr_train = nbr_train
        self.nbr_test = nbr_test
        self.batch_size = batch_size
        self.train = None
        self.test = None

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def output_is_sequence(self):
        return False

    @property
    def is_regression(self):
        return True

    def generate_sample(self):
        print ("Generate sample must be overwritten !")
        return -1, -1

    def generate_dataset(self, nbr_samples):
        samples = []
        outputs = []
        for i in range(nbr_samples):
            sample, groundtruth = self.generate_sample()
            samples.append(sample)
            outputs.append(groundtruth)
        return tf.concat(samples, 0), tf.concat(outputs, 0)

    @property
    def train_set(self):
        if self.train is None:
            self.train = self.generate_dataset(self.nbr_train)
        return tf.data.Dataset.from_tensor_slices(self.train).batch(self.batch_size)

    @property
    def test_set(self):
        if self.test is None:
            self.test = self.generate_dataset(self.nbr_test)
        return tf.data.Dataset.from_tensor_slices(self.test).batch(self.batch_size)

    @property
    def input_size(self):
        return len(self.train_set[0][0][0])

    @property
    def output_size(self):
        if not self.output_is_sequence:
            return len(self.train_set[1][0])
        else:
            return len(self.train_set[1][0][0])

    def build_sets(self):
        self.train_set
        self.test_set

class CopyInputDataset(Dataset):
    def __init__(self, nbr_train, nbr_test, batch_size, sequence_length,**kwargs):
        self.sequence_length = sequence_length
        super(CopyInputDataset, self).__init__(nbr_train, nbr_test, batch_size, **kwargs)

    @property
    def name(self):
        return self.__class__.__name__ + "_sequencelength_" + str(self.sequence_length)

    def generate_sample(self):
        true_n = np.random.randn()
        chain = np.concatenate([[true_n], np.random.randn(self.sequence_length - 1)])
        chain = chain.astype(np.float32)
        return tf.RaggedTensor.from_row_splits(values = np.reshape(chain, [-1, 1]), row_splits=[0,len(chain)]), tf.constant([[true_n]])

    @property
    def input_size(self):
        return 1

    @property
    def output_size(self):
        return 1

    @property
    def is_regression(self):
        return True

class DenoisingDataset(Dataset):
    def __init__(self, nbr_train, nbr_test, batch_size, nbr_relevant, length, no_useful_length, **kwargs):
        self.length = length
        self.no_useful_length = no_useful_length
        if self.no_useful_length == 0:
            self.no_useful_length = 1
        self.nbr_relevant = nbr_relevant
        super(DenoisingDataset, self).__init__(nbr_train, nbr_test, batch_size, **kwargs)

    @property
    def name(self):
        return self.__class__.__name__ + "_length_" + str(self.length) + "_nousefullength_" + str(self.no_useful_length) + "_nbrrelevant_" + str(self.nbr_relevant)

    def generate_sample(self,  length):
        inp = np.zeros((length+self.nbr_relevant-1, 2), dtype=np.float32)
        useful = np.random.choice(length-self.no_useful_length, self.nbr_relevant, replace = False)
        relevant_numbers = []
        for i in range(length):
            inp[i][0] = np.random.randn()
            if i in useful:
                relevant_numbers.append(inp[i][0])
                inp[i][1] = 1
        inp[length-1][0] = 0.0
        inp[length-1][1] = -1

        out = np.zeros((self.nbr_relevant, 1), dtype=np.float32)
        for cnt, rel_n in enumerate(relevant_numbers):
            out[cnt][0] = rel_n
        return tf.RaggedTensor.from_row_splits(values = inp, row_splits=[0, length+self.nbr_relevant-1]),\
               tf.RaggedTensor.from_row_splits(values = out, row_splits=[0, self.nbr_relevant])

    @property
    def input_size(self):
        return 2

    @property
    def output_size(self):
        return 1

    @property
    def is_regression(self):
        return True

    @property
    def output_is_sequence(self):
        return True

    def generate_dataset(self, nbr_samples, sequence_length):
        samples = []
        outputs = []
        for i in range(nbr_samples):
            sample, groundtruth = self.generate_sample(sequence_length)
            samples.append(sample)
            outputs.append(groundtruth)
        return tf.concat(samples, 0), tf.concat(outputs, 0)

    @property
    def train_set(self):
        if self.train is None:
            self.train = self.generate_dataset(self.nbr_train, self.length)
        return tf.data.Dataset.from_tensor_slices(self.train).batch(self.batch_size)

    @property
    def test_set(self):
        if self.test is None:
            self.test = self.generate_dataset(self.nbr_test, self.length)
        return tf.data.Dataset.from_tensor_slices(self.test).batch(self.batch_size)

class PermutedMnist(Dataset):
    def __init__(self, batch_size, black_pixels = 0):
        self.batch_size = batch_size
        self.black_pixels = black_pixels
        self.train = None
        self.test = None
        (self.x_train_i, self.y_train_i), (self.x_test_i, self.y_test_i) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
        self.x_train_i = np.reshape(self.x_train_i, [self.x_train_i.shape[0], -1, 1])
        self.x_test_i = np.reshape(self.x_test_i, [self.x_test_i.shape[0], -1, 1])
        self.y_train = tf.one_hot(self.y_train_i, 10)
        self.y_test = tf.one_hot(self.y_test_i, 10)
        np.random.seed(1)
        self.permut = np.random.permutation(784)

    def generate_dataset(self, training):
        if training:
            x, y = self.x_train_i, self.y_train
        else:
            x, y = self.x_test_i, self.y_test
        samples = []
        outputs = []
        for s,o in zip(x,y):
            s = (s / 255.) - 0.5
            s = s[self.permut]
            s = np.concatenate([s, np.zeros(shape = [self.black_pixels,1])],axis = 0)
            samples.append(tf.RaggedTensor.from_row_splits(values=s, row_splits=[0, 784 + self.black_pixels]))
            outputs.append([o])
        return tf.concat(samples, 0), tf.concat(outputs, 0)

    @property
    def train_set(self):
        if self.train is None:
            self.train = self.generate_dataset(True)
        return tf.data.Dataset.from_tensor_slices(self.train).batch(self.batch_size)

    @property
    def test_set(self):
        if self.test is None:
            self.test = self.generate_dataset(False)
        return tf.data.Dataset.from_tensor_slices(self.test).batch(self.batch_size)

    @property
    def input_size(self):
        return 1

    @property
    def output_size(self):
        return 10

    @property
    def name(self):
        return self.__class__.__name__ + "_black_pixels" + str(self.black_pixels)


    @property
    def is_regression(self):
        return False

    @property
    def output_is_sequence(self):
        return False


class RememberLinePMnist(Dataset):
    def __init__(self, batch_size, length, variable = False):
        self.batch_size = batch_size
        self.variable = variable
        self.length = length
        self.train = None
        self.test = None
        (self.x_train_i, self.y_train_i), (self.x_test_i, self.y_test_i) = tf.keras.datasets.mnist.load_data(
            path='mnist.npz')
        self.x_train_i = np.reshape(self.x_train_i, [self.x_train_i.shape[0], 28, 28])
        self.x_test_i = np.reshape(self.x_test_i, [self.x_test_i.shape[0], 28, 28])
        self.y_train = tf.one_hot(self.y_train_i, 10)
        self.y_test = tf.one_hot(self.y_test_i, 10)
        np.random.seed(1)
        self.permut = np.random.permutation(28)

    def generate_dataset(self, training):
        if training:
            x, y = self.x_train_i, self.y_train
        else:
            x, y = self.x_test_i, self.y_test
        if self.variable:
            gray = np.random.randint(self.length)
        else:
            gray = self.length
        samples = []
        outputs = []
        for s, o in zip(x, y):
            s = (s / 255.) - 0.5
            s = s[self.permut]
            s = np.concatenate([s, np.zeros(shape=[gray, 28])], axis=0)
            if self.variable:
                s[-1] = np.ones(shape=[28])
            samples.append(tf.RaggedTensor.from_row_splits(values=s, row_splits=[0, gray+28]))
            outputs.append([o])
        return tf.concat(samples, 0), tf.concat(outputs, 0)

    @property
    def train_set(self):
        if self.train is None:
            self.train = self.generate_dataset(True)
        return tf.data.Dataset.from_tensor_slices(self.train).batch(self.batch_size)

    @property
    def test_set(self):
        if self.test is None:
            self.test = self.generate_dataset(False)
        return tf.data.Dataset.from_tensor_slices(self.test).batch(self.batch_size)

    @property
    def input_size(self):
        return 28

    @property
    def output_size(self):
        return 10

    @property
    def name(self):
        return self.__class__.__name__ + "_length" + str(self.length) + "_variable_" + str(self.variable)

    @property
    def is_regression(self):
        return False

    @property
    def output_is_sequence(self):
        return False

