import tensorflow as tf
import math
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers
from tensorflow.keras.initializers import Constant, Initializer
from tensorflow.keras.initializers import (
    Identity as ID,
)  # Redefinition to avoid conflict with nengolib import

from nengolib.signal import Identity, cont2discrete
from nengolib.synapses import LegendreDelay
from scipy.special import legendre

def get_cells_list():
    return [tf.keras.layers.LSTMCell, tf.keras.layers.GRUCell, NBRC, BRC, GORU, LMU]

##### DEFINE THE NBRC ###########
class NBRC(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.state_size = [output_dim, output_dim, output_dim]
        super(NBRC, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):
        self.kernelz = self.add_weight(name="kz", shape=(input_shape[1], self.output_dim), dtype=tf.float32,
                                      initializer='glorot_uniform')
        self.kernelr = self.add_weight(name="kr", shape=(input_shape[1], self.output_dim), dtype=tf.float32,
                                      initializer='glorot_uniform')
        self.kernelh = self.add_weight(name="kh", shape=(input_shape[1], self.output_dim), dtype=tf.float32,
                                      initializer='glorot_uniform')

        self.memoryz = self.add_weight(name="mz", shape=(self.output_dim, self.output_dim), dtype=tf.float32,
                                      initializer='glorot_uniform')
        self.memoryr = self.add_weight(name="mr", shape=(self.output_dim, self.output_dim), dtype=tf.float32,
                                      initializer='glorot_uniform')

        self.br = self.add_weight(name="br", shape=(self.output_dim,), dtype = tf.float32, initializer='zeros')
        self.bz = self.add_weight(name="bz", shape=(self.output_dim,), dtype = tf.float32, initializer='zeros')

        super(NBRC, self).build(input_shape)

    def call(self, input, states):
        inp = input
        prev_out = states[0]
        z = tf.nn.sigmoid(tf.matmul(inp, self.kernelz) + tf.matmul(prev_out, self.memoryz) + self.bz)
        r = tf.nn.tanh(tf.matmul(inp, self.kernelr) + tf.matmul(prev_out, self.memoryr) + self.br)+1
        h = tf.nn.tanh(tf.matmul(inp, self.kernelh) + r * prev_out)
        output = (1.0 - z) * h + z * prev_out
        return output, [output, z, r]

    def get_config(self):
        return {"output_dim": self.output_dim}

############ DEFINE THE BRC #################
class BRC(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.state_size = [output_dim, output_dim, output_dim]
        super(BRC, self).__init__(output_dim, **kwargs)

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

        super(BRC, self).build(input_shape)

    def call(self, input, states):
        inp = input
        prev_out = states[0]
        r = tf.nn.tanh(tf.matmul(inp, self.kernelr) + prev_out * self.memoryr + self.br) + 1
        z = tf.nn.sigmoid(tf.matmul(inp, self.kernelz) + prev_out * self.memoryz + self.bz)
        output = z * prev_out + (1.0 - z) * tf.nn.tanh(tf.matmul(inp, self.kernelh) + r * prev_out)
        return output, [output, z, r]

    def get_config(self):
        return {"output_dim": self.output_dim}

############ DEFINE THE GORU #################
class GORU(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.state_size = output_dim
        self._capacity = int(math.log(self.output_dim, 2))
        super(GORU, self).__init__(output_dim, **kwargs)

    def modrelu(self, x, bias):
        """
        modReLU activation function
        """

        norm = tf.abs(x) + 0.001
        biased_norm = norm + bias
        magnitude = tf.nn.relu(biased_norm)
        phase = tf.sign(x)

        return phase * magnitude

    def generate_index_fft(self, s):
        """
        generate the index lists for goru to prepare orthogonal matrices
        and perform efficient rotations
        This function works for fft case
        """

        def ind_s(k):
            if k == 0:
                return np.array([[1, 0]])
            else:
                temp = np.array(range(2 ** k))
                list0 = [np.append(temp + 2 ** k, temp)]
                list1 = ind_s(k - 1)
                for i in range(k):
                    list0.append(np.append(list1[i], list1[i] + 2 ** k))
                return list0

        t = ind_s(int(math.log(s / 2, 2)))

        ind_exe = []
        for i in range(int(math.log(s, 2))):
            ind_exe.append(tf.constant(t[i]))

        ind_param = []
        for i in range(int(math.log(s, 2))):
            ind = np.array([])
            for j in range(2 ** i):
                ind = np.append(ind, np.array(
                    range(0, s, 2 ** i)) + j).astype(np.int32)

            ind_param.append(tf.constant(ind))

        return ind_exe, ind_param

    def fft_param(self, num_units):
        capacity = int(math.log(num_units, 2))

        cos_list = tf.concat([tf.cos(self.theta), tf.cos(self.theta)], axis=1)
        sin_list = tf.concat([tf.sin(self.theta), -tf.sin(self.theta)], axis=1)

        ind_exe, index_fft = self.generate_index_fft(num_units)

        v1 = tf.stack([tf.gather(cos_list[i, :], index_fft[i])
                       for i in range(capacity)])
        v2 = tf.stack([tf.gather(sin_list[i, :], index_fft[i])
                       for i in range(capacity)])
        return v1, v2, ind_exe

    def loop(self, h):
        v1, v2, ind = self.fft_param(self.output_dim)
        for i in range(self._capacity):
            diag = h * v1[i, :]
            off = h * v2[i, :]
            h = diag + tf.gather(off, ind[i], axis=1)
        return h

    def build(self, input_shape):
        phase_init = tf.random_uniform_initializer(-3.14, 3.14)
        capacity = int(math.log(self.output_dim, 2))
        self.theta = self.add_weight("theta", [capacity, self.output_dim // 2],
                                initializer=phase_init)

        input_matrix_init = tf.random_uniform_initializer(-0.01, 0.01)
        bias_init = tf.constant_initializer(2.)
        mod_bias_init = tf.constant_initializer(0.01)

        self.U = self.add_weight("U", [input_shape[1], self.output_dim * 3], dtype=tf.float32, initializer=input_matrix_init)

        self.W_r = self.add_weight("W_r", [self.output_dim, self.output_dim], dtype=tf.float32, initializer=input_matrix_init)
        self.W_g = self.add_weight("W_g", [self.output_dim, self.output_dim], dtype=tf.float32, initializer=input_matrix_init)

        self.bias_r = self.add_weight("bias_r", [self.output_dim], dtype=tf.float32, initializer=bias_init)
        self.bias_g = self.add_weight("bias_g", [self.output_dim], dtype=tf.float32)
        self.bias_c = self.add_weight("bias_c", [self.output_dim], dtype=tf.float32, initializer=mod_bias_init)

        super(GORU, self).build(input_shape)

    def call(self, input, states):
        Ux = tf.matmul(input, self.U)
        U_cx, U_rx, U_gx = tf.split(Ux, 3, axis=1)
        state = states[0]
        W_rh = tf.matmul(state, self.W_r)
        W_gh = tf.matmul(state, self.W_g)

        r_tmp = U_rx + W_rh + self.bias_r
        g_tmp = U_gx + W_gh + self.bias_g
        r = tf.sigmoid(r_tmp)
        g = tf.sigmoid(g_tmp)

        Unitaryh = self.loop(state)
        c = self.modrelu(r * Unitaryh + U_cx, self.bias_c)
        output = tf.multiply(g, state) + tf.multiply(1 - g, c)

        return output, [output]

    def get_config(self):
        return {"output_dim": self.output_dim}

############ DEFINE THE LEGENDRE MEMORY UNIT #################
class LMU(tf.keras.layers.Layer):
    class Legendre(Initializer):
        """
        Initializes weights using Legendre polynomials,
        leveraging scipy's legendre function. This may be used
        for the encoder and kernel initializers.
        """

        def __call__(self, shape, dtype=None):
            if len(shape) != 2:
                raise ValueError(
                    "Legendre initializer assumes shape is 2D; but shape=%s" % (shape,)
                )
            # TODO: geometric spacing might be useful too!
            return np.asarray(
                [legendre(i)(np.linspace(-1, 1, shape[1])) for i in range(shape[0])]
            )
    """
    Cell class for the LMU layer.
    This class processes one step within the whole time sequence input. Use the ``LMU``
    class to create a recurrent Keras layer to process the whole sequence. Calling
    ``LMU()`` is equivalent to doing ``RNN(LMUCell())``.
    Parameters
    ----------
    units : int
        The number of cells the layer will hold. This defines the dimensionality of the
        output vector.
    order : int
        The number of degrees in the transfer function of the LTI system used to
        represent the sliding window of history. With the default values (see the
        ``factory`` parameter), this parameter sets to the number of Legendre
        polynomials used to orthogonally represent the sliding window. This also
        defines the first dimensions of both the memory encorder and kernel as well as
        the the dimensions of the A and B matrices.
    theta : int
        The number of timesteps in the sliding window that are represented using the
        LTI system. In this context, the sliding window represents a dynamic range of
        data, of fixed size, that will be used to predict the value at the next time
        step. If this value is smaller than the size of the input sequence, only that
        number of steps will be represented in the A and B matrices at the time of
        prediction, however the entire sequence will still be processed in order for
        information to be projected to and from the hidden layer. This value is
        relative to a timestep of 1 second.
    method : string, optional
        The discretization method used to compute the A and B matrices. These matrices
        are used to map inputs onto the memory of the network.
    realizer : nengolib.signal, optional
        Determines what state space representation is being realized. This will be
        applied to the A and B matrices. Generally, unless you are training the A and B
        matrices, this should remain as its default.
    factory : nengolib.synapses, optional
        Determines what LTI system is being created. By default, this determines the
        A and B matrices. This can also be used to produce different realizations for
        the same LTI system. For example, using ``nengolib.synapses.PadeDelay``
        would give a rotation of ``nengolib.synapses.LegendreDelay``. In general, this
        allows you to swap out the dynamic primitive for something else entirely.
        (Default: ``nengolib.synapses.LegendreDelay``)
    trainable_input_encoders : bool, optional
        If True, the input encoders will be trained. This will allow for the encoders
        to learn what information is relevant to project from the input.
    trainable_hidden_encoders : bool, optional
        If True, the hidden encoders will be trained. This will allow for the encoders
        to learn what information is relevant to project from the hidden state.
    trainable_memory_encoders : bool, optional
        If True, the memory encoders will be trained. This will allow for the encoders
        to learn what information is relevant to project from the memory.
    trainable_input_kernel : bool, optional
        If True, the input kernel will be trained. This will allow for the kernel to
        learn to compute nonlinear functions across the memory.
    trainable_hidden_kernel : bool, optional
        If True, the hidden kernel will be trained. This will allow for the kernel to
        learn to compute nonlinear functions across the memory.
    trainable_memory_kernel : bool, optional
        If True, the memory kernel will be trained. This will allow for the kernel to
        learn to compute nonlinear functions across the memory.
    trainable_A : bool, optional
        If True, the A matrix will be trained via backpropagation, though this is
        generally not necessary as they can be derived.
    trainable_B : bool, optional
        If True, the B matrix will be trained via backpropagation, though this is
        generally not necessary as they can be derived.
    input_encoders_initializer : tf.keras.initializers.Initializer, optional
        The distribution for the input encoder weights initialization.
    hidden_encoders_initializer : tf.keras.initializers.Initializer, optional
        The distribution for the hidden encoder weights initialization.
    memory_encoders_initializer : tf.keras.initializers.Initializer, optional
        The distribution for the memory encoder weights initialization.
    input_kernel_initializer : tf.keras.initializers.Initializer, optional
        The distribution for the input kernel weights initialization.
    hidden_kernel_initializer : tf.keras.initializers.Initializer, optional
        The distribution for the hidden kernel weights initialization.
    memory_kernel_initializer : tf.keras.initializers.Initializer, optional
        The distribution for the memory kernel weights initialization.
    hidden_activation : string, optional
        The activation function to be used in the hidden component of the LMU.
    Attributes
    ----------
    state_size : tuple
        A tuple containing the units and order.
    output_size : int
        A duplicate of the units parameter.
    """

    def __init__(
        self,
        units,
        order = 256,
        theta = 1000,  # relative to dt=1
        method="zoh",
        realizer=Identity(),  # TODO: Deprecate?
        factory=LegendreDelay,  # TODO: Deprecate?
        trainable_input_encoders=True,
        trainable_hidden_encoders=True,
        trainable_memory_encoders=True,
        trainable_input_kernel=True,
        trainable_hidden_kernel=True,
        trainable_memory_kernel=True,
        trainable_A=False,
        trainable_B=False,
        input_encoders_initializer="lecun_uniform",
        hidden_encoders_initializer="lecun_uniform",
        memory_encoders_initializer=Constant(0),  # 'lecun_uniform',
        input_kernel_initializer="glorot_normal",
        hidden_kernel_initializer="glorot_normal",
        memory_kernel_initializer="glorot_normal",
        hidden_activation="tanh",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.units = units
        self.order = order
        self.theta = theta
        self.method = method
        self.realizer = realizer
        self.factory = factory
        self.trainable_input_encoders = trainable_input_encoders
        self.trainable_hidden_encoders = trainable_hidden_encoders
        self.trainable_memory_encoders = trainable_memory_encoders
        self.trainable_input_kernel = trainable_input_kernel
        self.trainable_hidden_kernel = trainable_hidden_kernel
        self.trainable_memory_kernel = trainable_memory_kernel
        self.trainable_A = trainable_A
        self.trainable_B = trainable_B

        self.input_encoders_initializer = initializers.get(input_encoders_initializer)
        self.hidden_encoders_initializer = initializers.get(hidden_encoders_initializer)
        self.memory_encoders_initializer = initializers.get(memory_encoders_initializer)
        self.input_kernel_initializer = initializers.get(input_kernel_initializer)
        self.hidden_kernel_initializer = initializers.get(hidden_kernel_initializer)
        self.memory_kernel_initializer = initializers.get(memory_kernel_initializer)

        self.hidden_activation = activations.get(hidden_activation)

        self._realizer_result = realizer(factory(theta=theta, order=self.order))
        self._ss = cont2discrete(
            self._realizer_result.realization, dt=1.0, method=method
        )
        self._A = self._ss.A - np.eye(order)  # puts into form: x += Ax
        self._B = self._ss.B
        self._C = self._ss.C
        assert np.allclose(self._ss.D, 0)  # proper LTI system

        # assert self._C.shape == (1, self.order)
        # C_full = np.zeros((self.units, self.order, self.units))
        # for i in range(self.units):
        #     C_full[i, :, i] = self._C[0]
        # decoder_initializer = Constant(
        #     C_full.reshape(self.units*self.order, self.units))

        # TODO: would it be better to absorb B into the encoders and then
        # initialize it appropriately? trainable encoders+B essentially
        # does this in a low-rank way

        # if the realizer is CCF then we get the following two constraints
        # that could be useful for efficiency
        # assert np.allclose(self._ss.B[1:], 0)  # CCF
        # assert np.allclose(self._ss.B[0], self.order**2)

        self.state_size = (self.units, self.order)
        self.output_size = self.units

    def build(self, input_shape):
        """
        Overrides the TensorFlow build function.
        Initializes all the encoders and kernels,
        as well as the A and B matrices for the
        LMUCell.
        """

        input_dim = input_shape[-1]

        # TODO: add regularizers

        self.input_encoders = self.add_weight(
            name="input_encoders",
            shape=(input_dim, 1),
            initializer=self.input_encoders_initializer,
            trainable=self.trainable_input_encoders,
        )

        self.hidden_encoders = self.add_weight(
            name="hidden_encoders",
            shape=(self.units, 1),
            initializer=self.hidden_encoders_initializer,
            trainable=self.trainable_hidden_encoders,
        )

        self.memory_encoders = self.add_weight(
            name="memory_encoders",
            shape=(self.order, 1),
            initializer=self.memory_encoders_initializer,
            trainable=self.trainable_memory_encoders,
        )

        self.input_kernel = self.add_weight(
            name="input_kernel",
            shape=(input_dim, self.units),
            initializer=self.input_kernel_initializer,
            trainable=self.trainable_input_kernel,
        )

        self.hidden_kernel = self.add_weight(
            name="hidden_kernel",
            shape=(self.units, self.units),
            initializer=self.hidden_kernel_initializer,
            trainable=self.trainable_hidden_kernel,
        )

        self.memory_kernel = self.add_weight(
            name="memory_kernel",
            shape=(self.order, self.units),
            initializer=self.memory_kernel_initializer,
            trainable=self.trainable_memory_kernel,
        )

        self.AT = self.add_weight(
            name="AT",
            shape=(self.order, self.order),
            initializer=Constant(self._A.T),  # note: transposed
            trainable=self.trainable_A,
        )

        self.BT = self.add_weight(
            name="BT",
            shape=(1, self.order),  # system is SISO
            initializer=Constant(self._B.T),  # note: transposed
            trainable=self.trainable_B,
        )

        self.built = True

    def call(self, inputs, states):
        """
        Overrides the TensorFlow call function.
        Contains the logic for one LMU step calculation.
        """

        h, m = states

        u = (
            K.dot(inputs, self.input_encoders)
            + K.dot(h, self.hidden_encoders)
            + K.dot(m, self.memory_encoders)
        )

        m = m + K.dot(m, self.AT) + K.dot(u, self.BT)

        h = self.hidden_activation(
            K.dot(inputs, self.input_kernel)
            + K.dot(h, self.hidden_kernel)
            + K.dot(m, self.memory_kernel)
        )

        return h, [h, m]

    def get_config(self):
        """
        Overrides the TensorFlow get_config function.
        Sets the config with the LMUCell parameters.
        """

        config = super().get_config()
        config.update(
            dict(
                units=self.units,
                order=self.order,
                theta=self.theta,
                method=self.method,
                factory=self.factory,
                trainable_input_encoders=self.trainable_input_encoders,
                trainable_hidden_encoders=self.trainable_hidden_encoders,
                trainable_memory_encoders=self.trainable_memory_encoders,
                trainable_input_kernel=self.trainable_input_kernel,
                trainable_hidden_kernel=self.trainable_hidden_kernel,
                trainable_memory_kernel=self.trainable_memory_kernel,
                trainable_A=self.trainable_A,
                trainable_B=self.trainable_B,
                input_encorders_initializer=self.input_encoders_initializer,
                hidden_encoders_initializer=self.hidden_encoders_initializer,
                memory_encoders_initializer=self.memory_encoders_initializer,
                input_kernel_initializer=self.input_kernel_initializer,
                hidden_kernel_initializer=self.hidden_kernel_initializer,
                memory_kernel_initializer=self.memory_kernel_initializer,
                hidden_activation=self.hidden_activation,
            )
        )

        return config
