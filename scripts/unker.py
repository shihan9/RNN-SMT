#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
python unker.py word2idx_file 15 num_files_to_be_unked file1, file2...
"""

import sys


def main():
    vcb = {}
    max_len = int(sys.argv[2])
    num_files = int(sys.argv[3])

    with open(sys.argv[1], 'r') as f:
        for line in f:
            word, index = line.split()
            vcb[word] = index

    for i in range(num_files):
        corpus = []
        with open(sys.argv[4 + i], 'r') as f:
            for line in f:
                words = line.split()
                for j in range(len(words)):
                    if words[j] not in vcb:
                        words[j] = '*UNK*'
                corpus.append(" ".join(words) + "\n")
        with open(sys.argv[4 + i] + '.mod', 'w') as f:
            f.writelines(corpus)


if __name__ == "__main__":
    main()
