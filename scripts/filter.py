#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
python filter.py file1 file2
"""

import sys


def main():
    maxlen = int(sys.argv[3])
    corpus1, corpus2 = [], []
    input_num, output_num = 0, 0

    with open(sys.argv[1], 'r') as f1, open(sys.argv[2], 'r') as f2:
        for line1, line2 in zip(f1, f2):
            words1, words2 = line1.split(), line2.split()
            if 1 <= len(words1) <= maxlen and 1 <= len(words2) <= maxlen:
                corpus1.append(" ".join(words1) + "\n")
                corpus2.append(" ".join(words2) + "\n")
                output_num += 1
            input_num += 1

    with open(sys.argv[1] + '.cleaned', 'w') as f1, open(
            sys.argv[2] + '.cleaned', 'w') as f2:
        f1.writelines(corpus1)
        f2.writelines(corpus2)
        print("Input Sentences:", input_num)
        print("Output Sentences:", output_num)


if __name__ == "__main__":
    main()
