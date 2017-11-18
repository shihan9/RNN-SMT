import numpy as np
import tensorflow as tf


class RNNencdec(object):

    def __init__(self, data, params):
        data = data
        batch_size = params.batch_size
        src_vocab_size = data.src_vocab_size
        tgt_vocab_size = data.tgt_vocab_size
        phase = params.phase
        rnn_size = params.rnn_size
        embed_size = params.embed_size
        alpha = params.alpha
        num_epoch = params.num_epoch

        tf.reset_default_graph()
        # encoder
        with tf.variable_scope("encoder"):
            src_inputs = tf.placeholder(tf.int32, [batch_size, None])
            # embedding lookup layer
            embeddings = tf.Variable(
                tf.random_normal((src_vocab_size, embed_size), stddev=.1))
            embedded = tf.nn.embedding_lookup(embeddings, src_inputs)
            # encoding
            src_seqlen = tf.placeholder(tf.int32, [batch_size])
            cell = tf.contrib.rnn.GRUCell(rnn_size)
            init_state = cell.zero_state(batch_size, tf.float32)
            outputs, final_state = tf.nn.dynamic_rnn(
                cell,
                embedded,
                sequence_length=src_seqlen,
                initial_state=init_state)

        # transform
        V = tf.Variable(tf.random_normal([batch_size, batch_size], stddev=0.1))
        Vb = tf.Variable(tf.random_normal([rnn_size], stddev=0.1))
        context = tf.nn.tanh(tf.matmul(V, final_state) + Vb)  # shape [BATCH_SIZE, RNN_SIZE]
        V_prime = tf.Variable(tf.random_normal([batch_size, batch_size], stddev=0.1))
        Vb_prime = tf.Variable(tf.random_normal([rnn_size], stddev=0.1))
        context = tf.nn.tanh(tf.matmul(V_prime, context) + Vb_prime)

        # decoder
        with tf.variable_scope("decoder"):
            tgt_inputs = tf.placeholder(tf.int32, [batch_size, None])
            # embedding lookup layer
            embeddings = tf.Variable(
                tf.random_normal((tgt_vocab_size, embed_size), stddev=.1))
            embedded = tf.nn.embedding_lookup(embeddings, tgt_inputs)
            # decoding
            tgt_seqlen = tf.placeholder(tf.int32, [batch_size])
            cell = tf.contrib.rnn.GRUCell(rnn_size)
            helper = tf.contrib.seq2seq.TrainingHelper(embedded, tgt_seqlen)
            decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, context)
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
            W = tf.Variable(tf.random_normal([rnn_size, tgt_vocab_size], stddev=.1))
            Wb = tf.Variable(tf.random_normal([tgt_vocab_size], stddev=.1))
            logits = tf.tensordot(outputs.rnn_output, W, [[2], [0]]) + Wb

        tgt_labels = tf.placeholder(tf.int32, [batch_size, None])
        masks = tf.sequence_mask(tgt_seqlen, None, dtype=tf.float32)

        cross_entropy = tf.contrib.seq2seq.sequence_loss(
            logits, tgt_labels, masks)
        opt = tf.train.AdamOptimizer(alpha)
        predictions = tf.argmax(logits, axis=2, output_type=tf.int32)
        accuracy = tf.contrib.metrics.accuracy(predictions, tgt_labels, masks)

        trainer = opt.minimize(cross_entropy)

        saver = tf.train.Saver()
        session = tf.Session()
        session.run(tf.global_variables_initializer())

        if phase == "TRAIN":
            # loss

            for batch in data.next_batch(batch_size, num_epoch):
                _, los = session.run(
                    [trainer, cross_entropy],
                    feed_dict={
                        src_inputs: batch[0],
                        tgt_inputs: batch[1],
                        tgt_labels: batch[2],
                        src_seqlen: batch[3],
                        tgt_seqlen: batch[4],
                    })
                print(los)

            saver.save(session, './model.ckpt')
        elif phase == 'DEV':
            saver.restore(session, './model.ckpt')

            num_batch = 0
            total_accu = 0
            for batch in data.next_batch(batch_size, num_epoch):
                tmp = session.run(
                    accuracy,
                    feed_dict={
                        src_inputs: batch[0],
                        tgt_inputs: batch[1],
                        tgt_labels: batch[2],
                        src_seqlen: batch[3],
                        tgt_seqlen: batch[4],
                    })
                total_accu += tmp
                num_batch += 1
            print(total_accu / num_batch * 100)
        elif phase == 'TEST':
            pass
