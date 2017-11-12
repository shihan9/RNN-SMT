"""
Project
Model 1
"""

import time
import sys
import helper
import tensorflow as tf
import numpy as np

WINDOW_SIZE = 13
BATCH_SIZE = 20
EMBEDDING_SIZE = 30
RNN_SIZE = 64
LEARNING_RATE = 1E-3


class Model1:
    def __init__(self, vocabulary_french_size, vocabulary_english_size):
        self.session = tf.Session()

        # input and target, french and english, keep probability
        self.batch_french = tf.placeholder(tf.int32, shape=[BATCH_SIZE, WINDOW_SIZE])
        self.batch_english_input = tf.placeholder(tf.int32, shape=[BATCH_SIZE, WINDOW_SIZE])
        self.batch_english_target = tf.placeholder(tf.int32, shape=[BATCH_SIZE, WINDOW_SIZE])
        self.batch_french_length = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
        self.batch_english_length = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
        self.keep_probability = tf.placeholder(tf.float32)

        # parameters
        self.weights = tf.Variable(tf.random_normal([RNN_SIZE, vocabulary_english_size], stddev=0.1))
        self.bias = tf.Variable(tf.random_normal([vocabulary_english_size], stddev=0.1))

        # encode
        with tf.variable_scope("encode"):
            # embedding
            self.embeddings_french = tf.Variable(tf.random_normal([vocabulary_french_size, EMBEDDING_SIZE], stddev=0.1))
            self.embedding_french = tf.nn.embedding_lookup(self.embeddings_french, self.batch_french)
            # dropout
            self.embedding_french = tf.nn.dropout(self.embedding_french, self.keep_probability)
            # GRU
            self.encode_cell = tf.contrib.rnn.GRUCell(RNN_SIZE)
            self.encode_previous_state = self.encode_cell.zero_state(BATCH_SIZE, tf.float32)
            self.encode_output, self.encode_next_state = tf.nn.dynamic_rnn(self.encode_cell, self.embedding_french,
                                                                           sequence_length=self.batch_french_length,
                                                                           initial_state=self.encode_previous_state)

        # process data from encoder for decoder
        # TODO: check the tile and reshape are correct
        self.V = tf.Variable(tf.random_normal([BATCH_SIZE, BATCH_SIZE], stddev=0.1))
        self.Vb = tf.Variable(tf.random_normal([RNN_SIZE], stddev=0.1))
        self.c = tf.nn.tanh(tf.matmul(self.V, self.encode_next_state) + self.Vb)  # shape [BATCH_SIZE, RNN_SIZE]
        self.c_tiled = tf.reshape(tf.tile(self.c, [1, WINDOW_SIZE]), [BATCH_SIZE, WINDOW_SIZE, RNN_SIZE])
        self.V_initial = tf.Variable(tf.random_normal([BATCH_SIZE, BATCH_SIZE], stddev=0.1))
        self.Vb_initial = tf.Variable(tf.random_normal([RNN_SIZE], stddev=0.1))
        self.decode_initial_state = tf.nn.tanh(tf.matmul(self.V_initial, self.c) + self.Vb_initial)

        # decode
        with tf.variable_scope("decode"):
            # embedding
            self.embeddings_english = tf.Variable(tf.random_normal([vocabulary_english_size, EMBEDDING_SIZE],
                                                                   stddev=0.1))
            self.embedding_english = tf.nn.embedding_lookup(self.embeddings_english, self.batch_english_input)
            self.embedding_STOP = tf.nn.embedding_lookup(self.embeddings_english,
                                                         tf.zeros(BATCH_SIZE, dtype=tf.int32))  # the STOP embedding
            # dropout
            self.embedding_english = tf.nn.dropout(self.embedding_english, self.keep_probability)
            # concat input and final state from encoder
            self.decode_input = tf.concat([self.embedding_english, self.c_tiled], 2)
            # GRU
            self.decode_cell = tf.contrib.rnn.GRUCell(RNN_SIZE)

            # define a dynamic decoder
            def loop_fn(window, cell_output, cell_state, loop_state):
                def get_next_input():
                    next_logits = tf.add(tf.matmul(cell_output, self.weights), self.bias)
                    prediction = tf.argmax(next_logits, axis=1)
                    return tf.nn.embedding_lookup(self.embeddings_english, prediction)

                # TODO: WINDOW_SIZE could be changed to self.batch_english_length
                elements_finished = (window >= WINDOW_SIZE)
                finished = tf.reduce_all(elements_finished)

                if cell_output is None:
                    next_input = self.embedding_STOP
                    next_cell_state = self.decode_initial_state
                else:
                    next_input = tf.cond(finished, lambda: self.embedding_STOP, get_next_input)
                    next_cell_state = cell_state

                next_input = tf.concat([next_input, self.c], 1)
                emit_output = cell_output
                next_loop_state = None

                return elements_finished, next_input, next_cell_state, emit_output, next_loop_state

            # TODO: maybe add switch, train with correct input, test with predict input
            # build a dynamic rnn
            # if is_train:
            #     self.decode_output, _ = tf.nn.dynamic_rnn(self.decode_cell, self.decode_input,
            #                                               sequence_length=self.batch_english_length,
            #                                               initial_state=self.decode_initial_state)
            # else:
            outputs_ta, _, _ = tf.nn.raw_rnn(self.decode_cell, loop_fn)
            self.decode_output = outputs_ta.stack()
            self.decode_output = tf.transpose(self.decode_output, [1, 0, 2])

        # logits
        self.logits = tf.tensordot(self.decode_output, self.weights, axes=[[2], [0]]) + self.bias

        # loss
        self.loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.batch_english_target,
                                                     tf.sequence_mask(self.batch_english_length, WINDOW_SIZE,
                                                                      dtype=tf.float32))

        # train
        self.train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

    # TODO: more than one epoch, maybe randomly choose input
    def train_rnn(self, batches_french, batches_english):
        # initial state
        previous_state = np.zeros([BATCH_SIZE, RNN_SIZE])

        for i in range(0, len(batches_french)):
            session_arguments = [self.encode_next_state, self.loss, self.train]
            feed_dict = {self.batch_french: batches_french[i],
                         self.batch_english_input: batches_english[i][:, 0:WINDOW_SIZE],
                         self.batch_english_target: batches_english[i][:, 1:WINDOW_SIZE+1],
                         self.batch_french_length: np.count_nonzero(batches_french[i], axis=1) + 1,
                         self.batch_english_length: np.count_nonzero(batches_english[i], axis=1) + 1,
                         self.encode_previous_state: previous_state,
                         self.keep_probability: 1}

            previous_state, output_loss, _ = self.session.run(session_arguments, feed_dict=feed_dict)

            # TODO: a better way to see the loss, maybe tensorboard
            if i % 1000 == 0:
                print(i, "loss", output_loss)

    def test_rnn(self, batches_french, batches_english):
        sum_probability = 0

        # initial state
        previous_state = np.zeros([BATCH_SIZE, RNN_SIZE])

        for i in range(0, len(batches_french)):
            session_arguments = [self.encode_next_state, self.logits]
            feed_dict = {self.batch_french: batches_french[i],
                         self.batch_english_input: batches_english[i][:, 0:WINDOW_SIZE],
                         self.batch_english_target: batches_english[i][:, 1:WINDOW_SIZE+1],
                         self.batch_french_length: np.count_nonzero(batches_french[i], axis=1) + 1,
                         self.batch_english_length: np.count_nonzero(batches_english[i], axis=1) + 1,
                         self.encode_previous_state: previous_state,
                         self.keep_probability: 1}

            previous_state, output_logits = self.session.run(session_arguments, feed_dict=feed_dict)

            batch_english_length = np.count_nonzero(batches_english[i], axis=1) + 1
            number_correct_english = 0
            predicate_english = np.argmax(output_logits, axis=2)
            for j in range(0, BATCH_SIZE):
                number_correct_english += np.sum(np.equal(predicate_english[j][0:batch_english_length[j]],
                                                 batches_english[i][:, 1:WINDOW_SIZE+1][j][0:batch_english_length[j]]))
            sum_probability += float(number_correct_english) / np.sum(batch_english_length)

        return sum_probability / len(batches_french)


if __name__ == "__main__":

    print("start Model 1")
    start_time = time.time()

    # read train and test files
    vocabulary_french_index, train_french = helper.read_train_french_data(sys.argv[1], WINDOW_SIZE, BATCH_SIZE)
    vocabulary_english_index, train_english = helper.read_train_english_data(sys.argv[2], WINDOW_SIZE, BATCH_SIZE)
    test_french = helper.read_test_french_data(sys.argv[3], vocabulary_french_index, WINDOW_SIZE, BATCH_SIZE)
    test_english = helper.read_test_english_data(sys.argv[4], vocabulary_english_index, WINDOW_SIZE, BATCH_SIZE)

    # init rnn
    model = Model1(len(vocabulary_french_index), len(vocabulary_english_index))
    model.session.run(tf.global_variables_initializer())

    # TODO: add saver
    # train rnn
    model.train_rnn(np.array(train_french), np.array(train_english))

    # run test
    accuracy = model.test_rnn(np.array(test_french), np.array(test_english))
    print("Test accuracy:", accuracy)

    print("--- %s seconds ---" % (time.time() - start_time))
