import numpy as np
import tensorflow as tf


class RNNencdec(object):

    def __init__(self, data, params):
        data = data
        sos = data.sos
        eos = data.eos
        src_vocab_size = data.src_vocab_size
        tgt_vocab_size = data.tgt_vocab_size
        batch_size = params.batch_size
        phase = params.phase
        rnn_size = params.rnn_size
        embed_size = params.embed_size
        alpha = params.alpha
        num_epoch = params.num_epoch

        src_inputs = tf.placeholder(tf.int32, [batch_size, None])
        tgt_inputs = tf.placeholder(tf.int32, [batch_size, None])
        tgt_labels = tf.placeholder(tf.int32, [batch_size, None])
        src_seqlen = tf.placeholder(tf.int32, [batch_size])
        tgt_seqlen = tf.placeholder(tf.int32, [batch_size])

        init_op = tf.random_normal_initializer
        tf.get_variable_scope().set_initializer(init_op)

        # encoder
        with tf.name_scope("encoder"):
            embeddings = tf.Variable(
                tf.random_normal((src_vocab_size, embed_size), stddev=.1))
            embedded = tf.nn.embedding_lookup(embeddings, src_inputs)

            cell = tf.contrib.rnn.GRUCell(rnn_size)
            init_state = cell.zero_state(batch_size, tf.float32)
            outputs, final_state = tf.nn.dynamic_rnn(
                cell,
                embedded,
                sequence_length=src_seqlen,
                initial_state=init_state)

            with tf.name_scope("context"):
                weights = tf.Variable(
                    tf.random_normal([batch_size, batch_size], stddev=0.1))
                biases = tf.Variable(tf.random_normal([rnn_size], stddev=0.1))
                context = tf.nn.tanh(tf.matmul(weights, final_state) + biases)

        # decoder
        with tf.name_scope("decoder"):
            with tf.name_scope("init_state"):
                weights = tf.Variable(
                    tf.random_normal([batch_size, batch_size], stddev=0.1))
                biases = tf.Variable(tf.random_normal([rnn_size], stddev=0.1))
                init_state = tf.nn.tanh(tf.matmul(weights, context) + biases)

            embeddings = tf.Variable(
                tf.random_normal((tgt_vocab_size, embed_size), stddev=.1))
            embedded = tf.nn.embedding_lookup(embeddings, tgt_inputs)

            cell = tf.contrib.rnn.GRUCell(rnn_size)
            output_layer = tf.layers.Dense(tgt_vocab_size, use_bias=True)
            output_layer.build([batch_size, None, rnn_size])

            max_time = tf.reduce_max(tgt_seqlen)

            if phase == 'TRAIN':
                tiled_context = tf.reshape(
                    tf.tile(context, [1, max_time]),
                    [batch_size, max_time, rnn_size])
                concat_inputs = tf.concat([embedded, tiled_context], -1)
                helper = tf.contrib.seq2seq.TrainingHelper(
                    concat_inputs, tgt_seqlen)
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell, helper, init_state)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
                logits = output_layer.apply(outputs.rnn_output)
                samples = None

                masks = tf.sequence_mask(tgt_seqlen, max_time, dtype=tf.float32)
                cross_entropy = tf.contrib.seq2seq.sequence_loss(
                    logits, tgt_labels, masks)
                opt = tf.train.AdamOptimizer(alpha)
                trainer = opt.minimize(cross_entropy)

            elif phase == 'DEV':
                # hack: append contexts
                def _embedding_fn(ids):
                    predicted_inputs = tf.nn.embedding_lookup(embeddings, ids)
                    new_inputs = tf.concat([predicted_inputs, context], -1)
                    return new_inputs
                    # return tf.nn.embedding_lookup(embeddings, ids)

                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    _embedding_fn, tf.fill([batch_size], sos), end_token=eos)

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell, helper, init_state, output_layer=output_layer)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder,
                    maximum_iterations=max_time * 2,
                    impute_finished=True)
                logits = outputs.rnn_output
                samples = outputs.sample_id

                distance = self._edit_distance(samples, tgt_labels, tgt_seqlen)

        saver = tf.train.Saver()
        session = tf.Session()

        if phase == "TRAIN":
            session.run(tf.global_variables_initializer())

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
            total_dist = 0
            for batch in data.next_batch(batch_size, num_epoch):
                tmp, _, _, _ = session.run(
                    [distance, self.out1, self.out2, self.out3],
                    feed_dict={
                        src_inputs: batch[0],
                        tgt_inputs: batch[1],
                        tgt_labels: batch[2],
                        src_seqlen: batch[3],
                        tgt_seqlen: batch[4],
                    })
                total_dist += tmp
                num_batch += 1
                print(num_batch)
                # if num_batch == 2:
                #     break
            print(total_dist / num_batch)
        elif phase == 'TEST':
            pass

    def _accuracy(self, logits, labels, masks):
        samples = tf.argmax(logits, axis=2, output_type=tf.int32)
        accuracy = tf.contrib.metrics.accuracy(samples, labels, masks)
        return accuracy

    def _edit_distance(self, samples, labels, seqlen):
        nonzero = tf.where(tf.not_equal(samples, 0))
        sparse_samples = tf.SparseTensor(nonzero,
                                         tf.gather_nd(samples, nonzero),
                                         tf.shape(samples, out_type=tf.int64))
        nonzero = tf.where(tf.not_equal(labels, 0))
        sparse_labels = tf.SparseTensor(nonzero,
                                        tf.gather_nd(labels, nonzero),
                                        tf.shape(labels, out_type=tf.int64))
        distance = tf.edit_distance(
            sparse_samples, sparse_labels, normalize=False)
        average = tf.reduce_mean(
            tf.divide(distance, tf.cast(seqlen, dtype=tf.float32)))
        self.out1 = tf.Print(samples, [samples[1]], summarize=20)
        self.out2 = tf.Print(labels, [labels[1]], summarize=20)
        self.out3 = tf.Print(distance, [distance[1]], summarize=20)
        return average


class RNNsearch(object):

    def __init__(self, data, params):
        data = data
        sos = data.sos
        eos = data.eos
        src_vocab_size = data.src_vocab_size
        tgt_vocab_size = data.tgt_vocab_size
        batch_size = params.batch_size
        phase = params.phase
        rnn_size = params.rnn_size
        embed_size = params.embed_size
        alpha = params.alpha
        num_epoch = params.num_epoch

        src_inputs = tf.placeholder(tf.int32, [batch_size, None])
        tgt_inputs = tf.placeholder(tf.int32, [batch_size, None])
        tgt_labels = tf.placeholder(tf.int32, [batch_size, None])
        src_seqlen = tf.placeholder(tf.int32, [batch_size])
        tgt_seqlen = tf.placeholder(tf.int32, [batch_size])

        init_op = tf.random_normal_initializer
        tf.get_variable_scope().set_initializer(init_op)

        # encoder
        with tf.name_scope("encoder"):
            embeddings = tf.Variable(
                tf.random_normal((src_vocab_size, embed_size), stddev=.1))
            embedded = tf.nn.embedding_lookup(embeddings, src_inputs)

            cell_fw = tf.contrib.rnn.GRUCell(rnn_size)
            cell_bw = tf.contrib.rnn.GRUCell(rnn_size)
            init_state_fw = cell_fw.zero_state(batch_size, tf.float32)
            init_state_bw = cell_bw.zero_state(batch_size, tf.float32)
            bi_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                embedded,
                sequence_length=src_seqlen,
                initial_state_fw=init_state_fw,
                initial_state_bw=init_state_bw)
            outputs = tf.concat(bi_outputs, -1)
            with tf.name_scope("context"):
                weights = tf.Variable(
                    tf.random_normal([batch_size, batch_size], stddev=0.1))
                biases = tf.Variable(tf.random_normal([rnn_size], stddev=0.1))
                context = tf.nn.tanh(tf.matmul(weights, final_state) + biases)

        with tf.name_scope("decoder"):
            with tf.name_scope("init_state"):
                weights = tf.Variable(
                    tf.random_normal([batch_size, batch_size], stddev=0.1))
                biases = tf.Variable(tf.random_normal([rnn_size], stddev=0.1))
                init_state = tf.nn.tanh(tf.matmul(weights, context) + biases)

            embeddings = tf.Variable(
                tf.random_normal((tgt_vocab_size, embed_size), stddev=.1))
            embedded = tf.nn.embedding_lookup(embeddings, tgt_inputs)

            cell = tf.contrib.rnn.GRUCell(rnn_size)
            output_layer = tf.layers.Dense(tgt_vocab_size, use_bias=True)
            output_layer.build([batch_size, None, rnn_size])

            max_time = tf.reduce_max(tgt_seqlen)

            if phase == 'TRAIN':
                tiled_context = tf.reshape(
                    tf.tile(context, [1, max_time]),
                    [batch_size, max_time, rnn_size])
                concat_inputs = tf.concat([embedded, tiled_context], -1)
                helper = tf.contrib.seq2seq.TrainingHelper(
                    concat_inputs, tgt_seqlen)
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell, helper, init_state)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
                logits = output_layer.apply(outputs.rnn_output)
                samples = None

                masks = tf.sequence_mask(tgt_seqlen, max_time, dtype=tf.float32)
                cross_entropy = tf.contrib.seq2seq.sequence_loss(
                    logits, tgt_labels, masks)
                opt = tf.train.AdamOptimizer(alpha)
                trainer = opt.minimize(cross_entropy)

            elif phase == 'DEV':
                # hack: append contexts
                def _embedding_fn(ids):
                    predicted_inputs = tf.nn.embedding_lookup(embeddings, ids)
                    new_inputs = tf.concat([predicted_inputs, context], -1)
                    return new_inputs
                    # return tf.nn.embedding_lookup(embeddings, ids)

                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    _embedding_fn, tf.fill([batch_size], sos), end_token=eos)

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell, helper, init_state, output_layer=output_layer)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder,
                    maximum_iterations=max_time * 2,
                    impute_finished=True)
                logits = outputs.rnn_output
                samples = outputs.sample_id

                distance = self._edit_distance(samples, tgt_labels, tgt_seqlen)

        saver = tf.train.Saver()
        session = tf.Session()

        if phase == "TRAIN":
            session.run(tf.global_variables_initializer())

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
            total_dist = 0
            for batch in data.next_batch(batch_size, num_epoch):
                tmp, _, _, _ = session.run(
                    [distance, self.out1, self.out2, self.out3],
                    feed_dict={
                        src_inputs: batch[0],
                        tgt_inputs: batch[1],
                        tgt_labels: batch[2],
                        src_seqlen: batch[3],
                        tgt_seqlen: batch[4],
                    })
                total_dist += tmp
                num_batch += 1
                print(num_batch)
                # if num_batch == 2:
                #     break
            print(total_dist / num_batch)
        elif phase == 'TEST':
            pass

    def _accuracy(self, logits, labels, masks):
        samples = tf.argmax(logits, axis=2, output_type=tf.int32)
        accuracy = tf.contrib.metrics.accuracy(samples, labels, masks)
        return accuracy

    def _edit_distance(self, samples, labels, seqlen):
        nonzero = tf.where(tf.not_equal(samples, 0))
        sparse_samples = tf.SparseTensor(nonzero,
                                         tf.gather_nd(samples, nonzero),
                                         tf.shape(samples, out_type=tf.int64))
        nonzero = tf.where(tf.not_equal(labels, 0))
        sparse_labels = tf.SparseTensor(nonzero,
                                        tf.gather_nd(labels, nonzero),
                                        tf.shape(labels, out_type=tf.int64))
        distance = tf.edit_distance(
            sparse_samples, sparse_labels, normalize=False)
        average = tf.reduce_mean(
            tf.divide(distance, tf.cast(seqlen, dtype=tf.float32)))
        self.out1 = tf.Print(samples, [samples[1]], summarize=20)
        self.out2 = tf.Print(labels, [labels[1]], summarize=20)
        self.out3 = tf.Print(distance, [distance[1]], summarize=20)
        return average
