import matplotlib.pyplot as plt
import numpy as np

from KMeans import k_means, compute_dist, load_dataset


def sum_squared_error(data_pt: np.ndarray, centroids: list, clusters: dict):
    dist_sq_list = list()

    for cluster_idx, data_idx in clusters.items():
        cluster_data = data_pt[data_idx]

        for data in cluster_data:
            dist = compute_dist(data, centroids[cluster_idx])
            dist_sq_list.append(np.square(dist))

    return sum(dist_sq_list)


def elbow_method(data_pt: np.ndarray):
    n = data_pt.shape[0]

    x_val = list()
    y_val = list()

    # for k in range(2, int(np.sqrt(n / 2))):
    for k in range(1, 10):
        centroids, clusters = k_means(data_pt, k)
        within_cluster_variance = sum_squared_error(data_pt, centroids, clusters)

        x_val.append(k)
        y_val.append(within_cluster_variance)

    plt.plot(x_val, y_val)
    plt.plot(x_val, y_val, 'or')
    plt.show()

    return dict(zip(x_val, y_val))


if __name__ == '__main__':
    data_pt = load_dataset('Dataset/google_review_ratings.csv', exclude_cols=[0], exclude_rows=[0])

    _elbow_data = elbow_method(data_pt)
