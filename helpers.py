import numpy as np


class Dataloader(object):

    def __init__(self, src_data, tgt_data, src_table, tgt_table, batch_size):
        self.batch_size = batch_size
        self.read_data(src_data, tgt_data)
        self.read_table(src_table, tgt_table)

    def read_data(self, src_data, tgt_data):
        self.src = []
        self.tgt = []
        with open(src_data, 'r') as f:
            for line in f:
                self.src.append(line.split())
        with open(tgt_data, 'r') as f:
            for line in f:
                self.tgt.append(line.split())
        self.size = len(self.src)
        self.win_size = len(self.src[0])

    def read_table(self, src_table, tgt_table):
        self.src_word2idx = {}
        self.tgt_word2idx = {}
        with open(src_table, 'r') as f:
            for line in f:
                word, index = line.split()
                self.src_word2idx[word] = index
        with open(tgt_table, 'r') as f:
            for line in f:
                word, index = line.split()
                self.tgt_word2idx[word] = index
        self.src_vocab_size = len(self.src_word2idx)
        self.tgt_vocab_size = len(self.tgt_word2idx)

    def next_batch(self):
        eos = np.ones([self.size, 1], dtype=np.int32)
        data = np.concatenate((eos, self.tgt), axis=1)  # appending STOPs to the front of inputs to decoder
        epoch_size = self.size // self.batch_size
        for i in range(0, epoch_size):
            subset = slice(i * self.batch_size,
                           i * self.batch_size + self.batch_size)
            seqlen1 = np.count_nonzero(self.src[subset, :], axis=1)
            seqlen2 = np.count_nonzero(
                data[subset, 1:1 + self.win_size], axis=1)
            yield self.src[subset, :], data[subset, 0:self.win_size], data[
                subset, 1:1 + self.win_size], seqlen1, seqlen2
