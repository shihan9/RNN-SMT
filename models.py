import numpy as np
import tensorflow as tf
from helpers import edit_distance, draw_attentions


class RNNsearch(object):

    def __init__(self, data, params):
        # parameters
        self.data = data
        self.sos = data.sos
        self.eos = data.eos
        self.src_vocab_size = data.src_vocab_size
        self.tgt_vocab_size = data.tgt_vocab_size
        self.src_vocab_table = data.src_idx2word
        self.tgt_vocab_table = data.tgt_idx2word
        self.batch_size = params.batch_size
        self.phase = params.phase
        self.rnn_size = params.rnn_size
        self.alignment_size = params.alignment_size
        self.embed_size = params.embed_size
        self.alpha = params.alpha
        self.num_epoch = params.num_epoch
        self.draw = 10

        # data fed into graph
        self.src_inputs = tf.placeholder(tf.int32, [self.batch_size, None])
        self.tgt_inputs = tf.placeholder(tf.int32, [self.batch_size, None])
        self.tgt_labels = tf.placeholder(tf.int32, [self.batch_size, None])
        self.src_seqlen = tf.placeholder(tf.int32, [self.batch_size])
        self.src_max_time = tf.reduce_max(self.src_seqlen)
        self.tgt_seqlen = tf.placeholder(tf.int32, [self.batch_size])
        self.tgt_max_time = tf.reduce_max(self.tgt_seqlen)

        # default initializer
        init_op = tf.random_normal_initializer(stddev=0.001)
        tf.get_variable_scope().set_initializer(init_op)

        # define graph
        self.loss, self.trainer, self.distance = self._graph()

        self.saver = tf.train.Saver()
        self.session = tf.Session()

    def train(self):
        assert self.phase == "TRAIN"
        self.session.run(tf.global_variables_initializer())
        num_batch = 0
        for batch in self.data.next_batch(self.batch_size, self.num_epoch):
            _, los = self.session.run(
                [self.trainer, self.loss],
                feed_dict={
                    self.src_inputs: batch[0],
                    self.tgt_inputs: batch[1],
                    self.tgt_labels: batch[2],
                    self.src_seqlen: batch[3],
                    self.tgt_seqlen: batch[4],
                })
            num_batch += 1
            print(los)

        self.saver.save(self.session, './model.ckpt')

    def dev(self):
        assert self.phase == 'DEV'
        self.saver.restore(self.session, './model.ckpt')
        num_batch = total_loss = 0
        for batch in self.data.next_batch(self.batch_size, 1):
            tmp = self.session.run(
                self.loss,
                feed_dict={
                    self.src_inputs: batch[0],
                    self.tgt_inputs: batch[1],
                    self.tgt_labels: batch[2],
                    self.src_seqlen: batch[3],
                    self.tgt_seqlen: batch[4],
                })
            total_loss += tmp
            num_batch += 1
        print(total_loss / num_batch)

    def test(self):
        assert self.phase == 'TEST'
        self.saver.restore(self.session, './model.ckpt')
        num_batch = total_dist = 0
        for batch in self.data.next_batch(self.batch_size, 1):
            if self.draw > 0:
                tmp, attentions, predictions, distances = self.session.run(
                    [
                        self.distance, self.attentions, self.predictions,
                        self.all_distances
                    ],
                    feed_dict={
                        self.src_inputs: batch[0],
                        self.tgt_inputs: batch[1],
                        self.tgt_labels: batch[2],
                        self.src_seqlen: batch[3],
                        self.tgt_seqlen: batch[4],
                    })
                self.draw = draw_attentions(self.draw, self.src_vocab_table,
                                            self.tgt_vocab_table, batch[0],
                                            predictions, attentions, distances)
            else:
                tmp = self.session.run(
                    self.distance,
                    feed_dict={
                        self.src_inputs: batch[0],
                        self.tgt_inputs: batch[1],
                        self.tgt_labels: batch[2],
                        self.src_seqlen: batch[3],
                        self.tgt_seqlen: batch[4],
                    })
            total_dist += tmp
            num_batch += 1
        print(total_dist / num_batch)

    def _graph(self):
        encoder_outputs, bw_first = self._encoder()
        return self._decoder(encoder_outputs, bw_first)

    def _encoder(self):
        with tf.name_scope("encoder"):
            embeddings = tf.Variable(
                tf.random_normal(
                    [self.src_vocab_size, self.embed_size], stddev=.1))
            embedded = tf.nn.embedding_lookup(embeddings, self.src_inputs)

            cell_fw = tf.contrib.rnn.GRUCell(
                self.rnn_size,
                kernel_initializer=tf.orthogonal_initializer(),
                bias_initializer=tf.zeros_initializer())
            cell_bw = tf.contrib.rnn.GRUCell(
                self.rnn_size,
                kernel_initializer=tf.orthogonal_initializer(),
                bias_initializer=tf.zeros_initializer())
            init_state_fw = cell_fw.zero_state(self.batch_size, tf.float32)
            init_state_bw = cell_bw.zero_state(self.batch_size, tf.float32)
            bi_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                embedded,
                sequence_length=self.src_seqlen,
                initial_state_fw=init_state_fw,
                initial_state_bw=init_state_bw)
            encoder_outputs = tf.concat(bi_outputs, -1)
            bw_first = bi_outputs[1][:, 0, :]
            return encoder_outputs, bw_first

    def _decoder(self, encoder_outputs, bw_first):
        with tf.name_scope("decoder"):
            init_state = tf.layers.dense(bw_first, self.rnn_size)

            embeddings = tf.Variable(
                tf.random_normal(
                    [self.tgt_vocab_size, self.embed_size], stddev=.1))
            embedded = tf.nn.embedding_lookup(embeddings, self.tgt_inputs)

            cell = tf.contrib.rnn.GRUCell(
                self.rnn_size,
                kernel_initializer=tf.orthogonal_initializer(),
                bias_initializer=tf.zeros_initializer())

            loss, trainer, distance = None, None, None
            if self.phase != 'TEST':
                logits, _ = self._dynamic_rnn_train(cell, embedded, init_state,
                                                    self.tgt_seqlen,
                                                    encoder_outputs)
                loss = tf.contrib.seq2seq.sequence_loss(logits, self.tgt_labels,
                                                        tf.sequence_mask(
                                                            self.tgt_seqlen,
                                                            self.tgt_max_time,
                                                            dtype=tf.float32))
                if self.phase == 'TRAIN':
                    opt = tf.train.AdamOptimizer(learning_rate=self.alpha)
                    grads_with_vars = opt.compute_gradients(loss)
                    g, v = zip(*grads_with_vars)
                    clipped_g, _ = tf.clip_by_global_norm(g, 1)
                    trainer = opt.apply_gradients(zip(clipped_g, v))

            if self.phase == 'TEST':
                sample_ids, _, attentions = self._dynamic_rnn_test(
                    cell, init_state, 2 * self.src_max_time,
                    tf.fill([self.batch_size], self.sos), self.eos, embeddings,
                    encoder_outputs)
                distance, all_distances = edit_distance(
                    sample_ids, self.tgt_labels, self.tgt_seqlen)
                if self.draw > 0:
                    self.attentions = attentions
                    self.all_distances = all_distances
                    self.predictions = sample_ids
            return loss, trainer, distance

    def _get_attention(self, state, hidden):
        with tf.name_scope("get_attention"):
            tiled_state = tf.reshape(
                tf.tile(state, [1, self.src_max_time]),
                [self.batch_size, self.src_max_time,
                 self.rnn_size])  # [batch_size, src_max_time, rnn_size]
            concated_states = tf.concat(
                [tiled_state,
                 hidden], -1)  # [batch_size, src_max_time, 3 * rnn_size]
            e_tilde = tf.layers.dense(
                concated_states,
                self.alignment_size,
                activation=tf.tanh,
                kernel_initializer=tf.random_normal_initializer(stddev=0.00001)
            )  # [batch_size, src_max_time, alignment_size]
            e = tf.squeeze(
                tf.layers.dense(
                    e_tilde, 1, kernel_initializer=tf.zeros_initializer())
            )  # [batch_size, src_max_time]
            attention = tf.nn.softmax(e)  # [batch_size, src_max_time]
            return attention

    def _get_context(self, attention, hidden):
        with tf.name_scope("get_context"):
            tiled_attention = tf.transpose(
                tf.reshape(
                    tf.tile(attention, [1, 2 * self.rnn_size]),
                    [self.batch_size, 2 * self.rnn_size, self.src_max_time]),
                [0, 2, 1])  # [batch_size, src_max_time, 2 * rnn_size]
            context = tf.reduce_sum(
                tiled_attention * hidden, axis=1)  # [batch_size, 2 * rnn_size]
            return context

    def _get_logits(self, outputs):
        logits = tf.layers.dense(outputs, self.tgt_vocab_size)
        return logits

    def _dynamic_rnn_train(self, cell, inputs, init_state, seqlen,
                           encoder_outputs):
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
            attention = self._get_attention(next_cell_state, encoder_outputs)
            context = self._get_context(attention, encoder_outputs)
            embedded = tf.cond(
                finished, lambda: tf.zeros([self.batch_size, self.embed_size]),
                lambda: inputs_ta.read(time))
            next_input = tf.cond(finished,
                                 lambda: tf.zeros([self.batch_size, self.embed_size + 2 * self.rnn_size], dtype=tf.float32),
                                 lambda: tf.concat([embedded, context], -1))

            next_loop_state = None
            return elements_finished, next_input, next_cell_state, emit_output, next_loop_state

        outputs_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
        outputs = tf.transpose(outputs_ta.stack(), [1, 0, 2])

        return outputs, final_state

    def _dynamic_rnn_test(self, cell, init_state, max_iterations, start_token,
                          end_token, embeddings, encoder_outputs):

        def loop_fn(time, cell_output, cell_state, loop_state):
            """
                loop_state: [attentions, last_finished]
            """
            if cell_output is not None:
                next_cell_state = cell_state
                sample_ids = tf.argmax(
                    self._get_logits(cell_output),
                    axis=-1,
                    output_type=tf.int32)
                emit_output = sample_ids
                next_loop_state = loop_state
            else:
                next_cell_state = init_state
                sample_ids = start_token
                emit_output = tf.constant(0, tf.int32)
                next_loop_state = [
                    tf.TensorArray(
                        dtype=tf.float32,
                        size=0,
                        dynamic_size=True,
                        element_shape=tf.TensorShape([self.batch_size, None])),
                    tf.fill([self.batch_size], False)
                ]

            # compute next input
            elements_finished = tf.equal(sample_ids, end_token)
            elements_finished = tf.logical_or(elements_finished,
                                              time >= max_iterations)
            attention = self._get_attention(next_cell_state, encoder_outputs)
            context = self._get_context(attention, encoder_outputs)
            embedded = tf.nn.embedding_lookup(embeddings, sample_ids)
            finished = tf.reduce_all(elements_finished)
            next_input = tf.cond(finished,
                                 lambda: tf.zeros([self.batch_size, self.embed_size + 2 * self.rnn_size], dtype=tf.float32),
                                 lambda: tf.concat([embedded, context], -1))

            # next_loop_state[1] is equal the finished vector maintained in raw_rnn,
            # so the attentions we generated has the same generated_tgt_max_time as sample_ids
            next_loop_state[1] = tf.logical_or(elements_finished,
                                               next_loop_state[1])
            next_loop_state = tf.cond(
                tf.reduce_all(next_loop_state[1]), lambda: next_loop_state,
                lambda: [next_loop_state[0].write(time, attention), next_loop_state[1]])

            return elements_finished, next_input, next_cell_state, emit_output, next_loop_state

        sample_ids_ta, final_state, loop_state = tf.nn.raw_rnn(cell, loop_fn)
        sample_ids = tf.transpose(sample_ids_ta.stack(),
                                  [1,
                                   0])  # [batch_size, generated_tgt_max_time]
        attentions = tf.transpose(
            loop_state[0].stack(),
            [1, 2, 0])  # [batch_size, src_max_time, generated_tgt_max_time]

        return sample_ids, final_state, attentions


class RNNencdec(object):

    def __init__(self, data, params):
        self.data = data
        self.sos = data.sos
        self.eos = data.eos
        self.src_vocab_size = data.src_vocab_size
        self.tgt_vocab_size = data.tgt_vocab_size
        self.batch_size = params.batch_size
        self.phase = params.phase
        self.rnn_size = params.rnn_size
        self.embed_size = params.embed_size
        self.alpha = params.alpha
        self.num_epoch = params.num_epoch

        self.src_inputs = tf.placeholder(tf.int32, [self.batch_size, None])
        self.tgt_inputs = tf.placeholder(tf.int32, [self.batch_size, None])
        self.tgt_labels = tf.placeholder(tf.int32, [self.batch_size, None])
        self.src_seqlen = tf.placeholder(tf.int32, [self.batch_size])
        self.src_max_time = tf.reduce_max(self.src_seqlen)
        self.tgt_seqlen = tf.placeholder(tf.int32, [self.batch_size])
        self.tgt_max_time = tf.reduce_max(self.tgt_seqlen)

        init_op = tf.random_normal_initializer(stddev=0.01)
        tf.get_variable_scope().set_initializer(init_op)

        self.loss, self.trainer, self.distance = self._graph()

        self.saver = tf.train.Saver()
        self.session = tf.Session()

    def train(self):
        assert self.phase == "TRAIN"
        self.session.run(tf.global_variables_initializer())

        for batch in self.data.next_batch(self.batch_size, self.num_epoch):
            _, los = self.session.run(
                [self.trainer, self.loss],
                feed_dict={
                    self.src_inputs: batch[0],
                    self.tgt_inputs: batch[1],
                    self.tgt_labels: batch[2],
                    self.src_seqlen: batch[3],
                    self.tgt_seqlen: batch[4],
                })
            print(los)

        self.saver.save(self.session, './model.ckpt')

    def dev(self):
        assert self.phase == 'DEV'
        self.saver.restore(self.session, './model.ckpt')
        num_batch = total_loss = 0
        for batch in self.data.next_batch(self.batch_size, 1):
            tmp, _, _, _ = self.session.run(
                self.loss,
                feed_dict={
                    self.src_inputs: batch[0],
                    self.tgt_inputs: batch[1],
                    self.tgt_labels: batch[2],
                    self.src_seqlen: batch[3],
                    self.tgt_seqlen: batch[4],
                })
            total_loss += tmp
            num_batch += 1
        print(total_loss / num_batch)

    def test(self):
        assert self.phase == 'TEST'
        self.saver.restore(self.session, './model.ckpt')
        num_batch = total_dist = 0
        for batch in self.data.next_batch(self.batch_size, 1):
            tmp = self.session.run(
                self.distance,
                feed_dict={
                    self.src_inputs: batch[0],
                    self.tgt_inputs: batch[1],
                    self.tgt_labels: batch[2],
                    self.src_seqlen: batch[3],
                    self.tgt_seqlen: batch[4],
                })
            total_dist += tmp
            num_batch += 1
        print(total_dist / num_batch)

    def _graph(self):
        encoder_outputs, final_state = self._encoder()
        return self._decoder(encoder_outputs, final_state)

    def _encoder(self):
        with tf.name_scope("encoder"):
            embeddings = tf.Variable(
                tf.random_normal(
                    (self.src_vocab_size, self.embed_size), stddev=.1))
            embedded = tf.nn.embedding_lookup(embeddings, self.src_inputs)

            cell = tf.contrib.rnn.GRUCell(self.rnn_size)
            init_state = cell.zero_state(self.batch_size, tf.float32)
            outputs, final_state = tf.nn.dynamic_rnn(
                cell,
                embedded,
                sequence_length=self.src_seqlen,
                initial_state=init_state)
            return outputs, final_state

    def _decoder(self, encoder_outputs, final_state):
        with tf.name_scope("decoder"):
            context = tf.layers.dense(
                final_state, self.rnn_size, activation=tf.nn.tanh)
            init_state = tf.layers.dense(
                context, self.rnn_size, activation=tf.nn.tanh)

            embeddings = tf.Variable(
                tf.random_normal(
                    (self.tgt_vocab_size, self.embed_size), stddev=.1))
            embedded = tf.nn.embedding_lookup(embeddings, self.tgt_inputs)

            cell = tf.contrib.rnn.GRUCell(self.rnn_size)
            output_layer = tf.layers.Dense(self.tgt_vocab_size, use_bias=True)
            output_layer.build([self.batch_size, None, self.rnn_size])

            loss, trainer, distance = None, None, None
            if self.phase != 'TEST':
                tiled_context = tf.reshape(
                    tf.tile(context, [1, self.tgt_max_time]),
                    [self.batch_size, self.tgt_max_time, self.rnn_size])
                concat_inputs = tf.concat([embedded, tiled_context], -1)
                helper = tf.contrib.seq2seq.TrainingHelper(
                    concat_inputs, self.tgt_seqlen)
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell, helper, init_state)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
                logits = output_layer.apply(outputs.rnn_output)

                masks = tf.sequence_mask(
                    self.tgt_seqlen, self.tgt_max_time, dtype=tf.float32)
                loss = tf.contrib.seq2seq.sequence_loss(logits, self.tgt_labels,
                                                        masks)
                if self.phase == 'TRAIN':
                    opt = tf.train.AdamOptimizer(self.alpha)
                    trainer = opt.minimize(loss)

            if self.phase == 'TEST':
                # hack: append contexts
                def _embedding_fn(ids):
                    predicted_inputs = tf.nn.embedding_lookup(embeddings, ids)
                    new_inputs = tf.concat([predicted_inputs, context], -1)
                    return new_inputs

                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    _embedding_fn,
                    tf.fill([self.batch_size], self.sos),
                    end_token=self.eos)

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell, helper, init_state, output_layer=output_layer)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder,
                    maximum_iterations=self.src_max_time * 2,
                    impute_finished=True)
                sample_ids = outputs.sample_id

                distance = edit_distance(sample_ids, self.tgt_labels,
                                         self.tgt_seqlen)

            return loss, trainer, distance
