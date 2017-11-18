import sys
import tensorflow as tf
import numpy as np


class RNNencdec(object):

    def __init__(self, data, rnn_size, batch_size, embed_size, alpha, num_epoch, phase):
        self.data = data
        self.batch_size = batch_size
        self.src_vocab_size = data.src_vocab_size
        self.tgt_vocab_size = data.tgt_vocab_size
        self.phase = phase
        self.rnn_size = rnn_size
        self.embed_size = embed_size
        self.alpha = alpha
        self.num_epoch = num_epoch
        self.session = tf.Session()
        self.opt = tf.train.AdamOptimizer(self.alpha)

    def _encoder(self):
        # encoder
        with tf.variable_scope("encoder"):
            self.src_inputs = tf.placeholder(tf.int32, [self.batch_size, None])
            # embedding lookup layer
            embeddings = tf.Variable(tf.random_normal((self.src_vocab_size, self.embed_size), stddev=.1))
            embedded = tf.nn.embedding_lookup(embeddings, self.src_inputs)
            # gru
            self.src_seqlen = tf.placeholder(tf.int32, [self.batch_size])
            cell = tf.contrib.rnn.GRUCell(self.rnn_size)
            init_state = cell.zero_state(self.batch_size, tf.float32)
            outputs, final_state = tf.nn.dynamic_rnn(
                cell, embedded, sequence_length=self.src_seqlen, initial_state=init_state)
            return final_state

    def _decoder(self, context):
        # decoder
        with tf.variable_scope("decoder"):
            self.tgt_inputs = tf.placeholder(tf.int32, [self.batch_size, None])
            # embedding lookup layer
            embeddings = tf.Variable(tf.random_normal((self.tgt_vocab_size, self.embed_size), stddev=.1))
            embedded = tf.nn.embedding_lookup(embeddings, self.tgt_inputs)
            # gru
            self.tgt_seqlen = tf.placeholder(tf.int32, [self.batch_size])
            cell = tf.contrib.rnn.GRUCell(self.rnn_size)
            # outputs, _ = tf.nn.dynamic_rnn(
            #     cell, embedded, sequence_length=self.tgt_seqlen, initial_state=context)
            helper = tf.contrib.seq2seq.TrainingHelper(embedded, self.tgt_seqlen)
            decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, context)
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
            weights = tf.Variable(tf.random_normal([self.rnn_size, self.tgt_vocab_size], stddev=.1))
            bias = tf.Variable(tf.random_normal([self.tgt_vocab_size], stddev=.1))
            logits = tf.tensordot(outputs.rnn_output, weights, [[2], [0]]) + bias
            return logits

    def _loss(self, logits, masks):
        cross_entropy = tf.contrib.seq2seq.sequence_loss(logits, self.tgt_labels, masks)
        return cross_entropy

    def _accuracy(self, logits, masks):
        predictions = tf.argmax(logits, axis=2, output_type=tf.int32)
        accuracy = tf.contrib.metrics.accuracy(predictions, self.tgt_labels, masks)
        return accuracy

    def train(self):
        self.tgt_labels = tf.placeholder(tf.int32, [self.batch_size, None])
        context = self._encoder()
        logits = self._decoder(context)
        masks = tf.sequence_mask(self.tgt_seqlen, None, dtype=tf.float32)
        loss = self._loss(logits, masks)
        # accuracy = self._accuracy(logits, tgt_labels, masks)
        trainer = self.opt.minimize(loss)
        self.session.run(tf.global_variables_initializer())

        for batch in self.data.next_batch(self.batch_size, self.num_epoch):
            _, los = self.session.run(
                [trainer, loss],
                feed_dict={
                    self.src_inputs: batch[0],
                    self.tgt_inputs: batch[1],
                    self.tgt_labels: batch[2],
                    self.src_seqlen: batch[3],
                    self.tgt_seqlen: batch[4],
                })
            print(los)

    def eval(self):
        pass
