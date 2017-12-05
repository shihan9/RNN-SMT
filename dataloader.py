import numpy as np
import random
import copy


class Dataloader(object):

    def __init__(self, src_data, tgt_data, src_table, tgt_table):
        self.read_data(src_data, tgt_data)
        self.read_table(src_table, tgt_table)
        self.sos = self.tgt_word2idx['*SOS*']
        self.eos = self.tgt_word2idx['*EOS*']
        self.pad = self.tgt_word2idx['*PAD*']

    def read_data(self, src_data, tgt_data):
        src = []
        tgt = []
        with open(src_data, 'r') as f:
            for line in f:
                src.append([int(x) for x in line.split()])
        with open(tgt_data, 'r') as f:
            for line in f:
                tgt.append([int(x) for x in line.split()])
        self.src = src
        self.tgt = tgt
        self.size = len(src)
        self.win_size = len(src[0])

    def read_table(self, src_table, tgt_table):
        self.src_word2idx = {}
        self.src_idx2word = {}
        self.tgt_word2idx = {}
        self.tgt_idx2word = {}
        with open(src_table, 'r') as f:
            for line in f:
                word, index = line.split()
                self.src_word2idx[word] = int(index)
        with open(tgt_table, 'r') as f:
            for line in f:
                word, index = line.split()
                self.tgt_word2idx[word] = int(index)
        for key, val in self.src_word2idx.items():
            self.src_idx2word[val] = key
        for key, val in self.tgt_word2idx.items():
            self.tgt_idx2word[val] = key
        self.src_vocab_size = len(self.src_word2idx)
        self.tgt_vocab_size = len(self.tgt_word2idx)

    def next_batch(self, batch_size, num_epoch, shuffle=False):
        idx_in_epoch = 0
        cur_epoch = 1
        indices = list(range(self.size))  # index of src and target
        if shuffle:
            random.shuffle(indices)

        while True:
            if idx_in_epoch + batch_size <= self.size:
                subset = slice(idx_in_epoch, idx_in_epoch + batch_size)
                subset = indices[subset]

                src_seq = [self.src[i][:] for i in subset]
                src_seqlen = list(map(lambda x: len(x) + 1, src_seq))
                src_maxlen = max(src_seqlen)
                self._padding(src_seq, 'SRC', src_maxlen)

                tgt_seq_inputs = [self.tgt[i][:] for i in subset]
                tgt_seq_labels = copy.deepcopy(tgt_seq_inputs)
                tgt_seqlen = list(map(lambda x: len(x) + 1, tgt_seq_inputs))
                tgt_maxlen = max(tgt_seqlen)

                self._padding(tgt_seq_inputs, 'TGT_INPUTS', tgt_maxlen)
                self._padding(tgt_seq_labels, 'TGT_LABELS', tgt_maxlen)

                idx_in_epoch += batch_size

            else:
                cur_epoch += 1
                if cur_epoch > num_epoch:
                    return
                # extract remained data
                subset = slice(idx_in_epoch, self.size)
                remained = indices[subset]

                if shuffle:
                    random.shuffle(indices)
                idx_in_epoch = batch_size - self.size + idx_in_epoch
                subset = slice(0, idx_in_epoch)
                subset = remained + indices[subset]

                src_seq = [self.src[i][:] for i in subset]
                src_seqlen = list(map(lambda x: len(x) + 1, src_seq))
                src_maxlen = max(src_seqlen)
                self._padding(src_seq, 'SRC', src_maxlen)

                tgt_seq_inputs = [self.tgt[i][:] for i in subset]
                tgt_seq_labels = copy.deepcopy(tgt_seq_inputs)
                tgt_seqlen = list(map(lambda x: len(x) + 1, tgt_seq_inputs))
                tgt_maxlen = max(tgt_seqlen)
                self._padding(tgt_seq_inputs, 'TGT_INPUTS', tgt_maxlen)
                self._padding(tgt_seq_labels, 'TGT_LABELS', tgt_maxlen)

            yield list(
                map(np.array, (src_seq, tgt_seq_inputs, tgt_seq_labels,
                               src_seqlen, tgt_seqlen)))

    # padding one batch with *EOS* *SOS* *PAD*
    def _padding(self, subset, flag, maxlen):
        for sentence in subset:
            if flag == 'SRC':
                sentence[:] = sentence + [self.eos] + [self.pad] * (
                    maxlen - len(sentence) - 1)
            elif flag == 'TGT_INPUTS':
                sentence[:] = [self.sos] + sentence + [self.pad] * (
                    maxlen - len(sentence) - 1)
            elif flag == 'TGT_LABELS':
                sentence[:] = sentence + [self.eos] + [self.pad] * (
                    maxlen - len(sentence) - 1)
            else:
                print("Wrong Flag")
