import numpy as np
from tabulate import tabulate


def load_dataset(file: str, exclude_cols: list, sep=','):
    data_pt = np.genfromtxt(file, delimiter=sep, skip_header=1)
    data_pt = np.delete(data_pt, obj=exclude_cols, axis=1)

    print(tabulate([['Dataset Size', data_pt.shape[0]], ['Instance Dimension', data_pt.shape[1]]], tablefmt='grid',
                   headers=['Dataset Summary', file]))

    return data_pt


def compute_dist(pt1: np.ndarray, pt2: np.ndarray):
    return np.sqrt(np.sum((pt1 - pt2) ** 2))


if __name__ == '__main__':
    print(load_dataset('Dataset/buddymove_holidayiq.csv', [0]))
