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
                    tf.random_normal([rnn_size, rnn_size], stddev=0.1))
                biases = tf.Variable(tf.random_normal([rnn_size], stddev=0.1))
                context = tf.nn.tanh(tf.matmul(final_state, weights) + biases)

        # decoder
        with tf.name_scope("decoder"):
            with tf.name_scope("init_state"):
                weights = tf.Variable(
                    tf.random_normal([rnn_size, rnn_size], stddev=0.1))
                biases = tf.Variable(tf.random_normal([rnn_size], stddev=0.1))
                init_state = tf.nn.tanh(tf.matmul(context, weights) + biases)

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
            for batch in data.next_batch(batch_size, 1):
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
        self.sos = data.sos
        self.eos = data.eos
        self.pad = data.pad
        self.src_vocab_size = data.src_vocab_size
        self.tgt_vocab_size = data.tgt_vocab_size
        self.batch_size = params.batch_size
        phase = params.phase
        self.rnn_size = params.rnn_size
        self.alignment_size = params.alignment_size
        self.embed_size = params.embed_size
        epsilon = params.epsilon
        rho = params.rho
        num_epoch = params.num_epoch
        self.maxout_size = params.maxout_size

        src_inputs = tf.placeholder(tf.int32, [self.batch_size, None])
        tgt_inputs = tf.placeholder(tf.int32, [self.batch_size, None])
        tgt_labels = tf.placeholder(tf.int32, [self.batch_size, None])
        src_seqlen = tf.placeholder(tf.int32, [self.batch_size])
        self.src_max_time = tf.reduce_max(src_seqlen)
        tgt_seqlen = tf.placeholder(tf.int32, [self.batch_size])
        self.tgt_max_time = tf.reduce_max(tgt_seqlen)
        keep_prob = tf.placeholder(tf.float32)

        init_op = tf.random_normal_initializer(stddev=0.01**2)
        tf.get_variable_scope().set_initializer(init_op)

        # encode
        with tf.name_scope("encode"):
            embeddings = tf.Variable(tf.random_normal([self.src_vocab_size, self.embed_size], stddev=.1))
            embedded = tf.nn.embedding_lookup(embeddings, src_inputs)
            embedded = tf.nn.dropout(embedded, keep_prob)

            cell_fw = tf.contrib.rnn.GRUCell(self.rnn_size, kernel_initializer=tf.orthogonal_initializer(), bias_initializer=tf.zeros_initializer())
            cell_bw = tf.contrib.rnn.GRUCell(self.rnn_size, kernel_initializer=tf.orthogonal_initializer(), bias_initializer=tf.zeros_initializer())
            init_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
            init_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)
            bi_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedded,
                                                                      sequence_length=src_seqlen,
                                                                      initial_state_fw=init_state_fw,
                                                                      initial_state_bw=init_state_bw)
            encoder_outputs = tf.concat(bi_outputs, -1)
            encoder_outputs_bw_first = bi_outputs[1][:, 0, :]

        # decode
        with tf.name_scope("decode"):
            embeddings = tf.Variable(tf.random_normal([self.tgt_vocab_size, self.embed_size], stddev=.1))
            embedded = tf.nn.embedding_lookup(embeddings, tgt_inputs)
            embedded = tf.nn.dropout(embedded, keep_prob)

            cell = tf.contrib.rnn.GRUCell(self.rnn_size, kernel_initializer=tf.orthogonal_initializer(), bias_initializer=tf.zeros_initializer())
            init_state = tf.layers.dense(encoder_outputs_bw_first, self.rnn_size)

            if phase != 'TEST':
                logits, _ = self.dynamic_rnn_train(cell, embedded, init_state, tgt_seqlen, encoder_outputs)
                loss = tf.contrib.seq2seq.sequence_loss(logits, tgt_labels, tf.sequence_mask(tgt_seqlen, self.tgt_max_time, dtype=tf.float32))
                if phase == 'TRAIN':
                    # opt = tf.train.AdadeltaOptimizer(rho=rho, epsilon=epsilon)
                    opt = tf.train.AdamOptimizer(learning_rate=1e-3)
                    # update = opt.minimize(loss)
                    grads_with_vars = opt.compute_gradients(loss)
                    g, v = zip(*grads_with_vars)
                    clipped_g, _ = tf.clip_by_global_norm(g, 1)
                    update = opt.apply_gradients(zip(clipped_g, v))

            if phase == 'TEST':
                sample_ids, _ = self.dynamic_rnn_test(cell, init_state, 2 * self.src_max_time, tf.fill([self.batch_size], self.sos), self.eos, embeddings, encoder_outputs)
                distance = self._edit_distance(sample_ids, tgt_labels, tgt_seqlen)

        saver = tf.train.Saver()
        session = tf.Session()

        if phase == "TRAIN":
            session.run(tf.global_variables_initializer())
            num_batch = 0
            for batch in data.next_batch(self.batch_size, num_epoch):
                _, los = session.run(
                    [update, loss],
                    feed_dict={
                        src_inputs: batch[0],
                        tgt_inputs: batch[1],
                        tgt_labels: batch[2],
                        src_seqlen: batch[3],
                        tgt_seqlen: batch[4],
                        keep_prob: 1.0
                    })
                num_batch += 1
                print(los)
                # if num_batch == 10:
                #     break

            saver.save(session, './model.ckpt')

        elif phase == 'DEV':
            saver.restore(session, './model.ckpt')
            num_batch = total_loss = 0
            for batch in data.next_batch(self.batch_size, 1):
                tmp = session.run(
                    loss,
                    feed_dict={
                        src_inputs: batch[0],
                        tgt_inputs: batch[1],
                        tgt_labels: batch[2],
                        src_seqlen: batch[3],
                        tgt_seqlen: batch[4],
                        keep_prob: 1.0
                    })
                total_loss += tmp
                num_batch += 1
                print(num_batch)
                # if num_batch == 2:
                #     break
            print(total_loss / num_batch)

        elif phase == 'TEST':
            saver.restore(session, './model.ckpt')
            num_batch = total_dist = 0
            for batch in data.next_batch(self.batch_size, 1):
                tmp = session.run(
                    distance,
                    feed_dict={
                        src_inputs: batch[0],
                        tgt_inputs: batch[1],
                        tgt_labels: batch[2],
                        src_seqlen: batch[3],
                        tgt_seqlen: batch[4],
                        keep_prob: 1.0
                    })
                total_dist += tmp
                num_batch += 1
                # if num_batch == 2:
                #     break
                # print(num_batch)
            print(total_dist / num_batch)

    def _get_context(self, state, hidden):
        with tf.name_scope("get_context"):
            tiled_state = tf.reshape(tf.tile(state, [1, self.src_max_time]), [self.batch_size, self.src_max_time, self.rnn_size])  # [batch_size, src_max_time, rnn_size]
            concated_states = tf.concat([tiled_state, hidden], -1)  # [batch_size, src_max_time, 3 * rnn_size]
            e_tilde = tf.layers.dense(concated_states, self.alignment_size, activation=tf.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.001**2))  # [batch_size, src_max_time, alignment_size]
            e = tf.squeeze(tf.layers.dense(e_tilde, 1, kernel_initializer=tf.zeros_initializer()))  # [batch_size, src_max_time]
            attention = tf.nn.softmax(e)  # [batch_size, src_max_time]
            tiled_attention = tf.transpose(tf.reshape(tf.tile(attention, [1, 2 * self.rnn_size]), [self.batch_size, 2 * self.rnn_size, self.src_max_time]), [0, 2, 1])  # [batch_size, src_max_time, 2 * rnn_size]
            context = tf.reduce_sum(tiled_attention * hidden, axis=1)  # [batch_size, 2 * rnn_size]
            return context

    # def _get_logits(self, state, embedded, context):
    #     t_tilde = tf.layers.dense(state, 2 * self.maxout_size) + tf.layers.dense(embedded, 2 * self.maxout_size) + tf.layers.dense(context, 2 * self.maxout_size)  # [batch_size, 2 * maxout_size]
    #     t = tf.reduce_max(tf.reshape(t_tilde, [self.batch_size, 2, self.maxout_size]), axis=1)  # [batch_size, maxout_size]
    #     logits = tf.nn.softmax(tf.layers.dense(t, self.tgt_vocab_size))  # [batch_size, tgt_vocab_size]
    #     return logits
    def _get_logits(self, outputs):
        logits = tf.layers.dense(outputs, self.tgt_vocab_size)
        return logits

    def dynamic_rnn_train(self, cell, inputs, init_state, seqlen, encoder_outputs):
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=self.tgt_max_time)
        inputs_ta = inputs_ta.unstack(tf.transpose(inputs, perm=[1, 0, 2]))

        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_output is not None:
                next_cell_state = cell_state
                emit_output = self._get_logits(cell_output)
            else:
                next_cell_state = init_state
                emit_output = tf.zeros([self.tgt_vocab_size])

            elements_finished = (time >= seqlen)
            finished = tf.reduce_all(elements_finished)
            context = self._get_context(next_cell_state, encoder_outputs)
            embedded = tf.cond(finished, lambda: tf.zeros([self.batch_size, self.embed_size]), lambda: inputs_ta.read(time))
            next_input = tf.cond(finished,
                                 lambda: tf.zeros([self.batch_size, self.embed_size + 2 * self.rnn_size], dtype=tf.float32),
                                 lambda: tf.concat([embedded, context], -1))

            # logits = self._get_logits(next_cell_state, embedded, context)
            next_loop_state = None
            return elements_finished, next_input, next_cell_state, emit_output, next_loop_state

        outputs_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
        outputs = tf.transpose(outputs_ta.stack(), [1, 0, 2])

        return outputs, final_state

    def dynamic_rnn_test(self, cell, init_state, max_iterations, start_token, end_token, embeddings, encoder_outputs):

        def loop_fn(time, cell_output, cell_state, loop_state):
            if cell_output is not None:
                next_cell_state = cell_state
                # sample_ids = loop_state
                # emit_output = sample_ids
                sample_ids = tf.argmax(self._get_logits(cell_output), axis=-1, output_type=tf.int32)
                emit_output = sample_ids
            else:
                next_cell_state = init_state
                sample_ids = start_token
                emit_output = tf.constant(0, tf.int32)

            # compute next input
            elements_finished = tf.equal(sample_ids, end_token)
            elements_finished = tf.logical_or(elements_finished, time >= max_iterations)
            context = self._get_context(next_cell_state, encoder_outputs)
            embedded = tf.nn.embedding_lookup(embeddings, sample_ids)
            finished = tf.reduce_all(elements_finished)
            next_input = tf.cond(finished,
                                 lambda: tf.zeros([self.batch_size, self.embed_size + 2 * self.rnn_size], dtype=tf.float32),
                                 lambda: tf.concat([embedded, context], -1))

            # logits = self._get_logits(next_cell_state, embedded, context)
            # next_loop_state = tf.argmax(logits, axis=-1, output_type=tf.int32)
            next_loop_state = None
            return elements_finished, next_input, next_cell_state, emit_output, next_loop_state

        sample_ids_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
        sample_ids = tf.transpose(sample_ids_ta.stack(), [1, 0])  # [batch_size, expected_tgt_time]
        return sample_ids, final_state

    def _edit_distance(self, samples, labels, seqlen):
        samples = tf.cast(samples, tf.int64)
        labels = tf.cast(labels, tf.int64)
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
        # self.out1 = tf.Print(samples, [samples], summarize=20)
        # self.out2 = tf.Print(labels, [labels], summarize=20)
        # self.out3 = tf.Print(distance, [distance], summarize=20)
        return average
