import timeit

import matplotlib.pyplot as plt
import numpy as np

from KMeans import k_means, compute_dist
from KMedoids import k_medoids, load_dataset


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

    # for k in range(2, int(np.floor(np.sqrt(n / 2)))):
    for k in range(2, 10):
        centroids, clusters = k_means(data_pt, k)
        within_cluster_variance = sum_squared_error(data_pt, centroids, clusters)

        x_val.append(k)
        y_val.append(within_cluster_variance)

    plt.plot(x_val, y_val, color='g')
    plt.plot(x_val, y_val, '+r')
    plt.show()

    return dict(zip(x_val, y_val))


def elbow_method_kmedoid(data_pt: np.ndarray):
    n = data_pt.shape[0]

    x_val = list()
    y_val = list()

    for k in range(2, 10):
        within_cluster_variance, _ = k_medoids(data_pt, k, use_abs_error=False)

        x_val.append(k)
        y_val.append(within_cluster_variance)

    plt.plot(x_val, y_val, color='y')
    plt.plot(x_val, y_val, 'or')
    plt.show()

    return dict(zip(x_val, y_val))


def time_comparison_graph(data_pt: np.ndarray):
    kmeans_time = list()
    kmedoids_time = list()
    x_val = list()

    for k in range(2, 11):
        x_val.append(k)

        start_kmeans = timeit.default_timer()
        _, _ = k_means(data_pt, k=k, visualize=False)
        stop_kmeans = timeit.default_timer()

        elapsed_time_kmeans = stop_kmeans - start_kmeans
        kmeans_time.append(elapsed_time_kmeans)

        start_kmedoids = timeit.default_timer()
        _, _ = k_medoids(data_pt, k=k, use_abs_error=False)
        stop_kmedoids = timeit.default_timer()

        elapsed_time_kmedoids = stop_kmedoids - start_kmedoids
        kmedoids_time.append(elapsed_time_kmedoids)

    # print(kmeans_time)
    # print(kmedoids_time)

    plt.plot(x_val, kmeans_time, color='g', label='K - Means')
    plt.plot(x_val, kmeans_time, 'or')

    plt.plot(x_val, kmedoids_time, color='b', label='K - Medoids')
    plt.plot(x_val, kmedoids_time, '^r')

    plt.xlabel('Value of K')
    plt.ylabel('Elapsed Time (Sec)')

    plt.legend()
    plt.title('Dataset : Weather Madrid (1997 - 2015)')

    plt.show()
    # plt.savefig('weather.png')



if __name__ == '__main__':
    data_pt = load_dataset('Dataset/weather_madrid_LEMD_1997_2015.csv', exclude_cols=[0, 22])

    # _elbow_data = elbow_method(data_pt)
    # elbow_method_kmedoid(data_pt)

    time_comparison_graph(data_pt)
