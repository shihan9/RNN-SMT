from collections import namedtuple
import time
from models import RNNencdec, RNNsearch
from helpers import Dataloader
import tensorflow as tf


def main():

    train_start = time.time()
    print("TRAINING")
    Params = namedtuple(
        'Params',
        ['batch_size', 'embed_size', 'rnn_size', 'alignment_size', 'maxout_size', 'epsilon', 'rho', 'phase', 'num_epoch'])

    data = Dataloader('data/hansard/train.fr', 'data/hansard/train.en',
                      'data/hansard/word2idx.fr', 'data/hansard/word2idx.en')
    # mparams = Params(20, 64, 64, 64, 64, 1e-6, 0.95, 'TRAIN', 1)
    # with tf.Graph().as_default():
    #     RNNsearch(data, mparams)

    train_end = time.time() - train_start
    print("--- %s seconds ---" % (train_end))

    print("TEST")
    data.read_data('data/hansard/dev.fr', 'data/hansard/dev.en')
    mparams = Params(20, 64, 64, 64, 64, 1e-6, 0.95, 'TEST', 1)
    with tf.Graph().as_default():
        RNNsearch(data, mparams)

    print("--- %s seconds ---" % (time.time() - train_end))


if __name__ == "__main__":
    main()
