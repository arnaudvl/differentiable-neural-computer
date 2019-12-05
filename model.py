import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, LSTM


class DNC(tf.keras.Model):

    def __init__(self,
                 output_dim: int,
                 memory_shape: tuple = (100, 20),
                 n_read: int = 3,
                 name: str = 'dnc'
                 ) -> None:
        """
        Initialize DNC object.

        Parameters
        ----------
        output_dim
            Size of output vector.
        memory_shape
            Shape of memory matrix (rows, cols).
        n_read
            Number of read heads.
        name
            Name of DNC.
        """
        super(DNC, self).__init__(name=name)

        # define output data size
        self.output_dim = output_dim  # Y

        # define size of memory matrix
        self.N, self.W = memory_shape  # N, W

        # define number of read heads
        self.R = n_read  # R

        # size of output vector from controller that defines interactions with memory matrix:
        # R read keys + R read strengths + write key + write strength + erase vector +
        # write vector + R free gates + allocation gate + write gate + R read modes
        self.interface_dim = self.R * self.W + 3 * self.W + 5 * self.R + 3  # I

        # neural net output = output of controller + interface vector with memory
        self.controller_dim = self.output_dim + self.interface_dim  # Y+I

        # initialize controller output and interface vector with gaussian normal
        self.output_v = tf.truncated_normal([1, self.output_dim], stddev=0.1)  # [1,Y]
        self.interface = tf.truncated_normal([1, self.interface_dim], stddev=0.1)  # [1,I]

        # initialize memory matrix with zeros
        self.M = tf.zeros(memory_shape)  # [N,W]

        # usage vector records which locations in the memory are used and which are free
        self.usage = tf.fill([self.N, 1], 1e-6)  # [N,1]

        # temporal link matrix L[i,j] records to which degree location i was written to after j
        self.L = tf.zeros([self.N, self.N])  # [N,N]

        # precedence vector determines degree to which a memory row was written to at t-1
        self.W_precedence = tf.zeros([self.N, 1])  # [N,1]

        # initialize R read weights and vectors and write weights
        self.W_read = tf.fill([self.N, self.R], 1e-6)  # [N,R]
        self.W_write = tf.fill([self.N, 1], 1e-6)  # [N,1]
        self.read_v = tf.fill([self.R, self.W], 1e-6)  # [R,W]

        # controller variables
        # initialize controller hidden state
        self.h = tf.Variable(tf.truncated_normal([1, self.controller_dim], stddev=0.1), name='dnc_h')  # [1,Y+I]
        self.c = tf.Variable(tf.truncated_normal([1, self.controller_dim], stddev=0.1), name='dnc_c')  # [1,Y+I]

        # initialise Dense and LSTM layers of the controller
        self.dense = Dense(self.W, activation=None)
        self.lstm = LSTM(
            self.controller_dim,
            return_sequences=False,
            return_state=True,
            name='dnc_controller'
        )

        # define and initialize weights for controller output and interface vectors
        self.W_output = tf.Variable(  # [Y+I,Y]
            tf.truncated_normal([self.controller_dim, self.output_dim], stddev=0.1),
            name='dnc_net_output_weights'
        )
        self.W_interface = tf.Variable(  # [Y+I,I]
            tf.truncated_normal([self.controller_dim, self.interface_dim], stddev=0.1),
            name='dnc_interface_weights'
        )

        # output y = v + W_read_out[r(1), ..., r(R)]
        self.W_read_out = tf.Variable(  # [R*W,Y]
            tf.truncated_normal([self.R * self.W, self.output_dim], stddev=0.1),
            name='dnc_read_vector_weights'
        )

    def content_lookup(self, key: tf.Tensor, strength: tf.Tensor) -> tf.Tensor:
        """
        Attention mechanism: content based addressing to read from and write to the memory.

        Params
        ------
        key
            Key vector emitted by the controller and used to calculate row-by-row
            cosine similarity with the memory matrix.
        strength
            Strength scalar attached to each key vector (1x1 or 1xR).

        Returns
        -------
        Similarity measure for each row in the memory used by the read heads for associative
        recall or by the write head to modify a vector in memory.
        """
        # The l2 norm applied to each key and each row in the memory matrix
        norm_mem = tf.nn.l2_normalize(self.M, 1)  # [N,W]
        norm_key = tf.nn.l2_normalize(key, 1)  # [1,W] for write or [R,W] for read

        # get similarity measure between both vectors, transpose before multiplication
        # write: [N*W]*[W*1] -> [N*1]
        # read: [N*W]*[W*R] -> [N,R]
        sim = tf.matmul(norm_mem, norm_key, transpose_b=True)
        return tf.nn.softmax(sim * strength, 0)  # [N,1] or [N,R]

    def allocation_weighting(self) -> tf.Tensor:
        """
        Memory needs to be freed up and allocated in a differentiable way.
        The usage vector shows how much each memory row is used.
        Unused rows can be written to. Usage of a row increases if
        we write to it and can decrease if we read from it, depending on the free gates.
        Allocation weights are then derived from the usage vector.

        Returns
        -------
        Allocation weights for each row in the memory.
        """
        # sort usage vector in ascending order and keep original indices of sorted usage vector
        sorted_usage, free_list = tf.nn.top_k(-1 * tf.transpose(self.usage), k=self.N)
        sorted_usage *= -1
        cumprod = tf.cumprod(sorted_usage, axis=1, exclusive=True)
        unorder = (1 - sorted_usage) * cumprod

        W_alloc = tf.zeros([self.N])
        I = tf.constant(np.identity(self.N, dtype=np.float32))

        # for each usage vec
        for pos, idx in enumerate(tf.unstack(free_list[0])):
            # flatten
            m = tf.squeeze(tf.slice(I, [idx, 0], [1, -1]))
            # add to allocation weight matrix
            W_alloc += m * unorder[0, pos]
        # return the allocation weighting for each row in memory
        return tf.reshape(W_alloc, [self.N, 1])

    def controller(self, x: tf.Tensor) -> None:
        # flatten input and pass through dense layer to avoid shape mismatch
        x = tf.reshape(x, [1, -1])
        x = self.dense(x)  # [1,W]

        # concatenate input with read vectors
        x_in = tf.expand_dims(Concatenate()([x, self.read_v], axis=0), axis=0)  # [1,R+1,W]

        # LSTM controller
        initial_state = [self.h, self.c]
        _, self.h, self.c = self.lstm(x_in, initial_state=initial_state)

    def partition_interface(self):
        # convert interface vector into a set of read write vectors
        partition = tf.constant([[0] * (self.R * self.W) + [1] * self.R +
                                 [2] * self.W + [3] + [4] * self.W + [5] * self.W +
                                 [6] * self.R + [7] + [8] + [9] * (self.R * 3)],
                                dtype=tf.int32)

        (k_read, b_read, k_write, b_write, erase, write_v, free_gates, alloc_gate,
         write_gate, read_modes) = tf.dynamic_partition(self.interface, partition, 10)

        # R read keys and strengths
        k_read = tf.reshape(k_read, [self.R, self.W])  # [R,W]
        b_read = 1 + tf.nn.softplus(tf.expand_dims(b_read, 0))  # [1,R]

        # write key, strength, erase and write vectors
        k_write = tf.expand_dims(k_write, 0)  # [1,W]
        b_write = 1 + tf.nn.softplus(tf.expand_dims(b_write, 0))  # [1,1]
        erase = tf.nn.sigmoid(tf.expand_dims(erase, 0))  # [1,W]
        write_v = tf.expand_dims(write_v, 0)  # [1,W]

        # the degree to which locations at read heads will be freed
        free_gates = tf.nn.sigmoid(tf.expand_dims(free_gates, 0))  # [1,R]

        # the fraction of writing that is being allocated in a new location
        alloc_gate = tf.reshape(tf.nn.sigmoid(alloc_gate), [1])  # 1

        # the amount of information to be written to memory
        write_gate = tf.reshape(tf.nn.sigmoid(write_gate), [1])  # 1

        # softmax distribution over the 3 read modes (forward, content lookup, backward)
        read_modes = tf.reshape(read_modes, [3, self.R])  # [3,R]
        read_modes = tf.nn.softmax(read_modes, axis=0)

        return (k_read, b_read, k_write, b_write, erase, write_v,
                free_gates, alloc_gate, write_gate, read_modes)

    def write(self,
              free_gates: tf.Tensor,
              alloc_gate: tf.Tensor,
              write_gate: tf.Tensor,
              k_write: tf.Tensor,
              b_write: tf.Tensor,
              erase: tf.Tensor,
              write_v: tf.Tensor
              ):
        # memory retention vector represents by how much each location will not be freed by the free gates
        retention = tf.reduce_prod(1 - free_gates * self.W_read, axis=1)
        retention = tf.reshape(retention, [self.N, 1])  # [N,1]

        # update usage vector which is used to dynamically allocate memory
        self.usage = (self.usage + self.W_write - self.usage * self.W_write) * retention

        # compute allocation weights using dynamic memory allocation
        W_alloc = self.allocation_weighting()  # [N,1]

        # apply content lookup for the write vector to figure out where to write to
        W_lookup = self.content_lookup(k_write, b_write)
        W_lookup = tf.reshape(W_lookup, [self.N, 1])  # [N,1]

        # define our write weights now that we know how much space to allocate for them and where to write to
        self.W_write = write_gate * (alloc_gate * W_alloc + (1 - alloc_gate) * W_lookup)

        # update memory matrix: erase memory and write using the write weights and vector
        self.M = (self.M * (1 - tf.matmul(self.W_write, erase)) + tf.matmul(self.W_write, write_v))

    def read(self,
             k_read: tf.Tensor,
             b_read: tf.Tensor,
             read_modes: tf.Tensor
             ):
        # update memory link matrix used later for the forward and backward read modes
        W_write_cast = tf.matmul(self.W_write, tf.ones([1, self.N]))  # [N,N]
        self.L = ((1 - W_write_cast - tf.transpose(W_write_cast)) * self.L +
                  tf.matmul(self.W_write, self.W_precedence, transpose_b=True))  # [N,N]
        self.L *= (tf.ones([self.N, self.N]) - tf.constant(np.identity(self.N, dtype=np.float32)))

        # update precedence vector which determines degree to which a memory row was written to at t-1
        self.W_precedence = ((1 - tf.reduce_sum(self.W_write, axis=0)) * self.W_precedence + self.W_write)

        # apply content lookup for the read vector(s) to figure out where to read from
        W_lookup = self.content_lookup(k_read, b_read)
        W_lookup = tf.reshape(W_lookup, [self.N, self.R])  # [N,R]

        # compute forward and backward read weights using the link matrix
        # forward weights recall information written in sequence and backward weights in reverse
        W_fwd = tf.matmul(self.L, self.W_read)  # [N,N]*[N,R] -> [N,R]
        W_bwd = tf.matmul(self.L, self.W_read, transpose_a=True)  # [N,R]

        # 3 modes: forward, backward and content lookup
        fwd_mode = read_modes[2] * W_fwd
        lookup_mode = read_modes[1] * W_lookup
        bwd_mode = read_modes[0] * W_bwd

        # read weights = backward + content lookup + forward mode weights
        self.W_read = bwd_mode + lookup_mode + fwd_mode  # [N,R]

        # create read vectors by applying read weights to memory matrix
        self.read_v = tf.transpose(tf.matmul(self.M, self.W_read, transpose_a=True))  # ([W,N]*[N,R])^T -> [R,W]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # update controller
        self.controller(x)

        # compute output and interface vectors
        self.output_v = tf.matmul(self.h, self.W_output)  # [1,Y+I] * [Y+I,Y] -> [1,Y]
        self.interface = tf.matmul(self.h, self.W_interface)  # [1,Y+I] * [Y+I,I] -> [1,I]

        # partition the interface vector
        (k_read, b_read, k_write, b_write, erase, write_v,
         free_gates, alloc_gate, write_gate, read_modes) = self.partition_interface()

        # write to memory
        self.write(free_gates, alloc_gate, write_gate, k_write, b_write, erase, write_v)

        # read from memory
        self.read(k_read, b_read, read_modes)

        # flatten read vectors and multiply them with W matrix before adding to controller output
        read_v_out = tf.matmul(tf.reshape(self.read_v, [1, self.R * self.W]),
                               self.W_read_out)  # [1,RW]*[RW,Y] -> [1,Y]

        # compute output
        y = self.output_v + read_v_out
        return y

    def fit(self):
        pass
