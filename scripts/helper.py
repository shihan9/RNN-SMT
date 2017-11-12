"""
HW6
bhuang13
Read data

Index of STOP is 0
"""

import numpy as np


def read_train_french_data(file_name, window_size, batch_size):
    vocabulary_index = {"STOP": 0}

    with open(file_name, "r") as train_file:
        windows_input = []

        # read line and generate window input
        for line in train_file:
            line_index = []
            words = line.split()

            # translate words to numbers
            for word in words:
                if word not in vocabulary_index:
                    vocabulary_index[word] = len(vocabulary_index)
                line_index.append(vocabulary_index[word])
            # padding STOP if it's less than window size
            while len(line_index) < window_size:
                line_index.append(0)
            # append to the window input
            windows_input.append(line_index)

        # split windows into batches
        batches_input = []
        batches_num = int(len(windows_input) / batch_size)
        for i in range(0, batches_num):
            batches_input.append([windows_input[j * batches_num + i] for j in range(batch_size)])

    return vocabulary_index, batches_input


def read_test_french_data(file_name, vocabulary_index, window_size, batch_size):
    with open(file_name, "r") as test_file:
        windows_input = []

        # read line and generate window input
        for line in test_file:
            line_index = []
            words = line.split()

            # translate words to numbers
            for word in words:
                if word not in vocabulary_index:
                    print("Test word cannot be found in vocabulary, ignore it:", word)
                else:
                    line_index.append(vocabulary_index[word])
            # padding STOP if it's less than window size
            while len(line_index) < window_size:
                line_index.append(0)
            # append to the window input
            windows_input.append(line_index)

        # split windows into batches
        batches_input = []
        batches_num = int(len(windows_input) / batch_size)
        for i in range(0, batches_num):
            batches_input.append([windows_input[j * batches_num + i] for j in range(batch_size)])

    return batches_input


def read_train_english_data(file_name, window_size, batch_size):
    vocabulary_index = {"STOP": 0}

    with open(file_name, "r") as train_file:
        windows_input = []

        # read line and generate window input
        for line in train_file:
            line_index = []
            words = line.split()

            # padding STOP at the first
            line_index.append(0)
            # translate words to numbers
            for word in words:
                if word not in vocabulary_index:
                    vocabulary_index[word] = len(vocabulary_index)
                line_index.append(vocabulary_index[word])
            # padding STOP if it's less than window size
            while len(line_index) < window_size + 1:
                line_index.append(0)
            # append to the window input
            windows_input.append(line_index)

        # split windows into batches
        batches_input = []
        batches_num = int(len(windows_input) / batch_size)
        for i in range(0, batches_num):
            batches_input.append([windows_input[j * batches_num + i] for j in range(batch_size)])

    return vocabulary_index, batches_input


def read_test_english_data(file_name, vocabulary_index, window_size, batch_size):
    with open(file_name, "r") as train_file:
        windows_input = []

        # read line and generate window input
        for line in train_file:
            line_index = []
            words = line.split()

            # padding STOP at the first
            line_index.append(0)
            # translate words to numbers
            for word in words:
                if word not in vocabulary_index:
                    print("Test word cannot be found in vocabulary, ignore it:", word)
                else:
                    line_index.append(vocabulary_index[word])
            # padding STOP if it's less than window size
            while len(line_index) < window_size + 1:
                line_index.append(0)
            # append to the window input
            windows_input.append(line_index)

        # split windows into batches
        batches_input = []
        batches_num = int(len(windows_input) / batch_size)
        for i in range(0, batches_num):
            batches_input.append([windows_input[j * batches_num + i] for j in range(batch_size)])

    return batches_input


def create_pseudo_attention(window_size):
    pseudo_attention = np.zeros([window_size, window_size], dtype=np.float32)
    for i in range(window_size):
        weights_sum = float(window_size)
        weights_count = window_size
        pseudo_attention[i][i] = 3.0 / window_size
        weights_sum -= 3.0
        weights_count -= 1
        if i - 1 >= 0:
            pseudo_attention[i][i - 1] = 1.5 / window_size
            weights_sum -= 1.5
            weights_count -= 1
        if i + 1 < window_size:
            pseudo_attention[i][i + 1] = 1.5 / window_size
            weights_sum -= 1.5
            weights_count -= 1

        weights_left = weights_sum / weights_count
        for j in range(window_size):
            if pseudo_attention[i][j] == 0:
                pseudo_attention[i][j] = weights_left / window_size

    # pseudo_weights = np.zeros([WINDOW_SIZE, WINDOW_SIZE], dtype=np.float32)
    # for i in range(WINDOW_SIZE):
    #     for j in range(WINDOW_SIZE):
    #         if i == j:
    #             pseudo_weights[i][j] = 3.0 / (WINDOW_SIZE + 2)
    #         else:
    #             pseudo_weights[i][j] = 1.0 / (WINDOW_SIZE + 2)

    return pseudo_attention
