#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
python indexer.py word2idx_file num_files_to_be_indexed file1, file2...
"""

import sys


def main():
    vcb = {}
    num_files = int(sys.argv[2])

    with open(sys.argv[1], 'r') as f:
        for line in f:
            word, index = line.split()
            vcb[word] = index

    for i in range(num_files):
        corpus = []
        with open(sys.argv[3 + i], 'r') as f:
            for line in f:
                words = line.split()
                for j in range(len(words)):
                    words[j] = vcb[words[j]]
                corpus.append(" ".join(words) + "\n")
        with open(sys.argv[3 + i] + '.idx', 'w') as f:
            f.writelines(corpus)


if __name__ == "__main__":
    main()
