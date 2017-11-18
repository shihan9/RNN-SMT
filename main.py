import sys
import time
from models import RNNencdec
from helpers import Dataloader


def main():

    embed_size = 30
    rnn_size = 64
    batch_size = 20
    alpha = 1e-3
    batch_size = 20
    num_epoch = 1

    print("start")
    start_time = time.time()

    data = Dataloader('data/hansard/train.fr', 'data/hansard/train.en', 'data/hansard/word2idx.fr', 'data/hansard/word2idx.en')
    train_model = RNNencdec(data, rnn_size, batch_size, embed_size, alpha, num_epoch, phase='TRAIN')
    train_model.train()

    data.read_data('data/hansard/dev.fr', 'data/hansard/dev.en')
    dev_model = RNNencdec(data, 'DEV', rnn_size, embed_size, alpha)
    accuracy = train_model.train()
    print("accuracy:", accuracy)

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
