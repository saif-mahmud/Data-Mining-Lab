import timeit

import numpy as np
from tabulate import tabulate

from KMedoids import load_dataset
from Visualization import Cluster_viz


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


def k_means(data_pt: np.ndarray, k: int, visualize=False):
    centroids = data_pt[init_centroid(data_pt, k)]

    i = 0

    if visualize:
        viz = Cluster_viz(data_pt)

    while True:
        i = i + 1

        _clusters = np.zeros(data_pt.shape[0])
        clusters = {label: [] for label in range(k)}
        for data_idx in range(data_pt.shape[0]):
            min_idx = cluster_assignment(data_pt, data_idx, centroids, clusters)
            clusters[min_idx].append(data_idx)
            _clusters[data_idx] = min_idx

        # print(centroids)
        # print(update_centroid(data_pt, clusters))

        if np.array_equal(centroids, update_centroid(data_pt, clusters)):
            break

        centroids = update_centroid(data_pt, clusters)

        table = []

        for title, indices in clusters.items():
            table.append([title, len(indices)])

        print('\nIteration -', i)
        print(tabulate(table, headers=['Cluster', '# of Members'], tablefmt="fancy_grid"))
        if visualize:
            viz.visualize_iteration(i, _clusters)


if __name__ == '__main__':
    # data_pt = load_dataset('Dataset/wine.data', exclude_cols=[0])
    data_pt = load_dataset('Dataset/google_review_ratings.csv', exclude_cols=[0], exclude_rows=[0])

    # print(data_pt)
    start = timeit.default_timer()
    k_means(data_pt, k=5, visualize=True)
    stop = timeit.default_timer()

    print('\nTotal Time Elepsed (Sec) :', stop - start)
