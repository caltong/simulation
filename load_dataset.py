import numpy as np
import os


def get_one_file_data(file_path):
    one_file_data = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            one_file_data.append(float(line))
    return one_file_data


def get_one_file_data_and_compress(file_path):
    one_file_data = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            one_file_data.append(float(line))
        length = len(one_file_data)
        rate = int(length / 128)
        compressed_one_file_data = []
        for i in range(128):
            s = sum(one_file_data[i * rate:i * rate + 128]) / rate
            compressed_one_file_data.append(s)

    return compressed_one_file_data


def load_dataset():
    x = []
    y = []
    path = 'simulation'
    dirs = os.listdir(path)
    for subdir in dirs:
        y.append(int(subdir[-1]) - 1)  # 1-10 to 0-9
        subdir = os.path.join(path, subdir)
        files = os.listdir(subdir)
        data_single = []
        for file in files:
            data_single.append(get_one_file_data(os.path.join(subdir, file)))
        x.append(data_single)
    x = np.array(x).reshape((-1, 1280))
    y = np.array(y)
    return x, y


def load_test_dataset():
    x = []
    y = []
    path = os.path.join('experiment')
    dirs = os.listdir(path)
    for subdir in dirs:
        y.append(int(subdir[-1]) - 1)  # 1-10 to 0-9
        subdir = os.path.join(path, subdir)
        files = os.listdir(subdir)
        data_single = []
        for file in files:
            data_single.append(get_one_file_data_and_compress(os.path.join(subdir, file)))
        x.append(data_single)
    x = np.array(x).reshape((-1, 1280))
    y = np.array(y)
    return x, y
