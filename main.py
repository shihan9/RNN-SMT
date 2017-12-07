from collections import namedtuple
import time
from models import RNNencdec, RNNsearch
from dataloader import Dataloader
import tensorflow as tf


def main():

    train_start = time.time()
    print("TRAINING")
    Params = namedtuple(
        'Params',
        ['batch_size', 'embed_size', 'rnn_size', 'alignment_size', 'alpha', 'phase', 'num_epoch'])

    data = Dataloader('data/hansard/train.fr', 'data/hansard/train.en',
                      'data/hansard/word2idx.fr', 'data/hansard/word2idx.en')
    mparams = Params(50, 128, 256, 256, 1e-3, 'TRAIN', 5)
    with tf.Graph().as_default():
        RNNsearch(data, mparams).train()

    train_end = time.time() - train_start
    print("--- %s seconds ---" % (train_end))

    print("TEST")
    data.read_data('data/hansard/dev.fr', 'data/hansard/dev.en')
    mparams = Params(50, 128, 256, 256, 1e-3, 'TEST', 1)
    with tf.Graph().as_default():
        RNNsearch(data, mparams).test()

    print("--- %s seconds ---" % (time.time() - train_end))


if __name__ == "__main__":
    main()
