#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from collections import Counter


def main():
    vcb_size = int(sys.argv[1])
    lang = sys.argv[2]
    num_files = int(sys.argv[3])
    counter = Counter()
    vcb = {'*PAD*': 0, '*EOS*': 1, '*UNK*': 2}

    for i in range(num_files):
        with open(sys.argv[4 + i], 'r') as f:
            for line in f:
                words = line.split()
                counter.update(words)

    for word, count in counter.most_common(vcb_size):
        vcb[word] = len(vcb)

    with open('word2idx.' + lang, 'w') as f1, open('idx2word.' + lang,
                                                   'w') as f2:
        f1.writelines(
            [" ".join([item[0], str(item[1])]) + "\n" for item in vcb.items()])
        f2.writelines(
            [" ".join([str(item[1]), item[0]]) + "\n" for item in vcb.items()])


if __name__ == "__main__":
    main()
