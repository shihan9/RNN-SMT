import tensorflow as tf


def edit_distance(self, samples, labels, seqlen):
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
    return average
