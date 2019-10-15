from pprint import pprint

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


def init_centroid(data_pt: np.ndarray, k: int):
    # np.random.seed(54)
    centroid_idx = np.random.randint(low=0, high=data_pt.shape[0], size=k)

    return centroid_idx


def cluster_assignment(data_pt: np.ndarray, data_idx: int, centroids: np.ndarray, clusters: dict):
    distance = dict()

    for centroid_idx in clusters.keys():
        distance[centroid_idx] = compute_dist(data_pt[data_idx], centroids[centroid_idx])

    # print('Distance :', distance)
    min_idx = min(distance, key=distance.get)

    return min_idx


def update_centroid(data_pt: np.ndarray, clusters: dict):
    centroids_updated = list()

    for data_idx in clusters.values():
        cl_data = data_pt[data_idx]
        mean = np.mean(cl_data, axis=0)

        centroids_updated.append(mean)

    return centroids_updated


def k_means(data_pt: np.ndarray, k: int):
    centroids = data_pt[init_centroid(data_pt, k)]

    i = 0

    while True:
        i = i + 1
        print('Iteration -', i)

        clusters = {label: [] for label in range(k)}
        for data_idx in range(data_pt.shape[0]):
            min_idx = cluster_assignment(data_pt, data_idx, centroids, clusters)
            clusters[min_idx].append(data_idx)

        # print(centroids)
        # print(update_centroid(data_pt, clusters))

        if np.array_equal(centroids, update_centroid(data_pt, clusters)):
            break

        centroids = update_centroid(data_pt, clusters)

        for title, indices in clusters.items():
            print(title, ':', len(indices))

        print('\n')


if __name__ == '__main__':
    data_pt = load_dataset('Dataset/buddymove_holidayiq.csv', exclude_cols=[0])

    # print(data_pt)
    k_means(data_pt, k=4)
    pprint('---###---')
