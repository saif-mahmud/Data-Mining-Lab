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
    centroid_idx = np.random.randint(low=0, high=data_pt.shape[0], size=k)

    return centroid_idx


def cluster_assignment(data_pt: np.ndarray, data_idx: int, centroids: np.ndarray, clusters: dict):
    distance = dict()

    for centroid_idx in clusters.keys():
        distance[centroid_idx] = compute_dist(data_pt[data_idx], centroids[centroid_idx])

    # print('Distance :', distance)
    min_idx = min(distance, key=distance.get)

    return min_idx


def k_means(data_pt: np.ndarray, k: int):
    centroids = data_pt[init_centroid(data_pt, k)]
    clusters = {label: [] for label in range(k)}

    # for i in range(5):
    for data_idx in range(data_pt.shape[0]):
        min_idx = cluster_assignment(data_pt, data_idx, centroids, clusters)
        clusters[min_idx].append(data_idx)

    # pprint(clusters)

    # print(data_pt[clusters[0]])

    # for label in clusters.keys():
        


if __name__ == '__main__':
    data_pt = load_dataset('Dataset/buddymove_holidayiq.csv', exclude_cols=[0])

    # print(data_pt)
    k_means(data_pt, k=4)
