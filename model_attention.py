import numpy as np
import tensorflow as tf

class RNNsearch(object):

    def __init__(self, data, params):
        data = data
        sos = data.sos
        eos = data.eos
        pad = data.pad
        src_vocab_size = data.src_vocab_size
        tgt_vocab_size = data.tgt_vocab_size
        batch_size = params.batch_size
        phase = params.phase
        rnn_size = params.rnn_size
        alignment_size = params.alignment_size
        embed_size = params.embed_size
        alpha = params.alpha  # TODO learning rate ?
        num_epoch = params.num_epoch

        src_inputs = tf.placeholder(tf.int32, [batch_size, None])
        tgt_inputs = tf.placeholder(tf.int32, [batch_size, None])
        tgt_labels = tf.placeholder(tf.int32, [batch_size, None])
        src_seqlen = tf.placeholder(tf.int32, [batch_size])
        src_max_time = tf.reduce_max(src_seqlen)
        tgt_seqlen = tf.placeholder(tf.int32, [batch_size])
        tgt_max_time = tf.reduce_max(src_seqlen)
        keep_prob = tf.placeholder(tf.float32)

        # parameters for predicition
        weights = tf.Variable(tf.random_normal([rnn_size, tgt_vocab_size], stddev=0.1))
        biases = tf.Variable(tf.random_normal([tgt_vocab_size], stddev=0.1))

        init_op = tf.random_normal_initializer
        tf.get_variable_scope().set_initializer(init_op)

        # encode
        with tf.variable_scope("encode"):
            # embedding
            embeddings = tf.Variable(tf.random_normal([src_vocab_size, embed_size], stddev=.1))
            embedded = tf.nn.embedding_lookup(embeddings, src_inputs)
            # dropout
            embedded = tf.nn.dropout(embedded, keep_prob)
            # GRU
            cell_fw = tf.contrib.rnn.GRUCell(rnn_size)
            cell_bw = tf.contrib.rnn.GRUCell(rnn_size)
            init_state_fw = cell_fw.zero_state(batch_size, tf.float32)
            init_state_bw = cell_bw.zero_state(batch_size, tf.float32)
            bi_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedded,
                                                                      sequence_length=src_seqlen,
                                                                      initial_state_fw=init_state_fw,
                                                                      initial_state_bw=init_state_bw)
            encode_outputs = tf.concat(bi_outputs, -1)
            encode_outputs_bw_first = bi_outputs[1][0]  # TODO double check
        
        # decode
        with tf.variable_scope("decode"):
            # embedding
            embeddings = tf.Variable(tf.random_normal([tgt_vocab_size, embed_size], stddev=.1))
            embedded = tf.nn.embedding_lookup(embeddings, tgt_inputs)
            # dropout
            embedded = tf.nn.dropout(embedded, keep_prob)
            # GRU
            cell = tf.contrib.rnn.GRUCell(rnn_size)

            # define function to compute attention
            init_weights = tf.Variable(tf.random_normal([rnn_size, rnn_size]), stddev=0.1)
            init_biases = tf.Variable(tf.random_normal([rnn_size]), stddev=0.1)
            va_weights = tf.Variable(tf.random_normal([alignment_size, 1]), stddev=0.1)
            va_biases = tf.Variable(tf.random_normal([1]), stddev=0.1)
            wa_weights = tf.Variable(tf.random_normal([rnn_size, alignment_size]), stddev=0.1)
            wa_biases = tf.Variable(tf.random_normal([alignment_size]), stddev=0.1)
            ua_weights = tf.Variable(tf.random_normal([2 * rnn_size, alignment_size]), stddev=0.1)
            ua_biases = tf.Variable(tf.random_normal([alignment_size]), stddev=0.1)
            def compute_attention(state):
                ws = tf.matmul(state, wa_weights) + wa_biases  # [batch_size, alignment_size]
                tiled_ws = tf.reshape(tf.tile(ws, [1, src_max_time]), [batch_size, src_max_time, alignment_size]) # [batch_size, src_max_time, alignment_size]
                uh = tf.tensordot(encode_outputs, ua_weights, [[2], [0]]) + ua_biases)  # [batch_size, src_max_time, alignment_size]
                e = tf.tensordot(tf.tanh(tiled_ws + uh), va_weights, [[2],[0]]) + va_biases  # TODO [batch_size, src_max_time] or [batch_size, src_max_time, 1]
                attention = tf.nn.softmax(e, dim=-1)  # [batch_size, src_max_time]
                return attention

            if phase == 'TRAIN':
                # define loop function for training
                inputs_ta = tf.TensorArray(dtype=tf.float32, size=tgt_max_time)
                inputs_ta = inputs_ta.unstack(tf.transpose(embedded, perm=[1, 0, 2]))  # TODO double check [tgt_max_time, batch_size, embed_size]
                def loop_fn_train(time, cell_output, cell_state, loop_state):

                    def get_next_input(state):
                        inputs = inputs_ta.read(time)
                        attentions = compute_attention(state)
                        tiled_attentions = tf.transpose(tf.reshape(tf.tile(attention, [1, 2 * rnn_size]), [batch_size, rnn_size, src_max_time]), [0, 2, 1])
                        context = encode_outputs * tiled_attention  # element-wise multiply, [batch_size, src_max_time, 2 * rnn_size]
                        context = tf.reduce_sum(context, axis=1)
                        return tf.concat([inputs, context], 1)

                    # elements_finished
                    elements_finished = (time >= tgt_seqlen)
                    finished = tf.reduce_all(elements_finished)
                    # next_cell_state
                    if cell_output is None:
                        next_cell_state = tf.matmul(encode_outputs_bw_first, init_weights) + init_biases  # [batch_size, rnn_size]
                    else:
                        next_cell_state = cell_state
                    # next_input
                    next_input = tf.cond(finished, lambda: tf.zeros([batch_size, embed_size+2*rnn_size], dtype=tf.float32),
                                                   lambda: get_next_input(next_cell_state))  # TODO double check that pad and zero have the same effect
                    # emit_output and next_loop_state
                    emit_output = cell_output
                    next_loop_state = None
                    return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

                decode_outputs_ta, _, _ = tf.rnn.raw_rnn(cell, loop_fn_train)
                decode_outputs = decode_outputs_ta.stack()
                decode_outputs = tf.transpose(decode_outputs, [1, 0, 2])

                logits = tf.tensordot(decode_outputs, weights, axes[[2], [0]]) + biases
                loss = tf.contrib.seq2seq.sequence_loss(logits, tgt_labels, tf.sequence_mask(tgt_seqlen, tgt_max_time, dtype=tf.float32)
                trainer = tf.train.AdamOptimizer(alpha).minimize(loss)

            elif phase == 'TEST':
                index_pad = tf.fill([batch_size], pad)
                # define loop function for testing
                def loop_fn_test(time, cell_output, cell_state, loop_state):

                    def get_next_input(prediction, state):
                        inputs = tf.nn.embedding_lookup(embeddings, prediction)
                        attentions = compute_attention(state)
                        tiled_attentions = tf.transpose(tf.reshape(tf.tile(attention, [1, 2 * rnn_size]), [batch_size, rnn_size, src_max_time]), [0, 2, 1])
                        context = encode_outputs * tiled_attention  # element-wise multiply, [batch_size, src_max_time, 2 * rnn_size]
                        context = tf.reduce_sum(context, axis=1)
                        return tf.concat([inputs, context], 1) 

                    # elements_finished
                    elements_finished = (time >= 2 * src_max_time)
                    finished = tf.reduce_all(elements_finished)
                    # next_cell_state
                    if cell_output is None:
                        next_cell_state = tf.matmul(encode_outputs_bw_first, init_weights) + init_biases  # [batch_size, rnn_size]
                    else:
                        next_cell_state = cell_state
                    # next_input
                    if is finished:
                        next_input = tf.zeros([batch_size, embed_size+2*rnn_size], dtype=tf.float32)  # TODO double check that pad and zero have the same effect
                    elif time > 0:
                        next_logits = tf.matmul(cell_output, weights) + biases
                        next_predictions = tf.argmax(next_logits, axis=1)
                        next_predictions = (next_predictions * tf.logical_not(loop_state)) + (loop_state * pad)  # use pad as input based on loop_state
                        next_input = get_next_input(next_predictions, next_cell_state)
                    else:  # time == 0
                        next_input = get_next_input(sos, next_cell_state)
                    # emit_output
                    emit_output = cell_output
                    # next_loop_state
                    if next_loop_state is None:
                        next_loop_state = tf.zeros([batch_size, dtype=tf.bool])
                    else:
                        next_loop_state = tf.logical_or(next_loop_state, tf.equal(next_predictions, index_pad))
                        if is not finished and is tf.reduce_all(next_loop_state):  # stop earlier
                            elements_finished = (time == time)
                    return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

                decode_outputs_ta, _, _ = tf.rnn.raw_rnn(cell, loop_fn_test)
                decode_outputs = decode_outputs_ta.stack()
                decode_outputs = tf.transpose(decode_outputs, [1, 0, 2])

                logits = tf.tensordot(decode_outputs, weights, axes[[2], [0]]) + biases
                # TODO distance

            
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
            pass
            
        elif phase == 'TEST':
            saver.restore(session, './model.ckpt')
            pass
