import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def edit_distance(samples, labels, seqlen):
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
    distance = tf.edit_distance(sparse_samples, sparse_labels, normalize=False)
    average = tf.reduce_mean(
        tf.divide(distance, tf.cast(seqlen, dtype=tf.float32)))
    return average, distance


def draw_attentions(num_draw, src_table, tgt_table, src_idx, tgt_idx, attentions, distances):
    for i in range(len(distances)):
        if num_draw > 0 and distances[i] <= 1:
            src = np.trim_zeros(src_idx[i], 'b')
            tgt = np.trim_zeros(tgt_idx[i], 'b')
            src_len = len(src)
            tgt_len = len(tgt)
            if src_len != tgt_len or src_len < 11:
                continue
            atts = attentions[i, :src_len, :tgt_len]
            plt.imshow(atts, 'gray')

            plt.xticks(range(tgt_len), [tgt_table[x].decode('utf-8') for x in tgt], rotation=90)
            plt.yticks(range(src_len), [src_table[y].decode('utf-8') for y in src])
            plt.gca().xaxis.tick_top()
            plt.savefig('data/pics/' + str(num_draw) + '.png', bbox_inches='tight')
            num_draw -= 1

    return num_draw
