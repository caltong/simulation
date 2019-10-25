import numpy as np
import os


def get_one_file_data(file_path):
    one_file_data = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            one_file_data.append(float(line))
    return one_file_data


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