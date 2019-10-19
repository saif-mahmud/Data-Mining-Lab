from copy import deepcopy

import numpy as np
from tabulate import tabulate

from Visualize import Cluster_viz


def load_dataset(file: str, exclude_cols: list, sep=','):
    data_pt = np.genfromtxt(file, delimiter=sep, skip_header=1)
    data_pt = np.delete(data_pt, obj=exclude_cols, axis=1)

    print(tabulate([['Dataset Size', data_pt.shape[0]], ['Instance Dimension', data_pt.shape[1]]], tablefmt='grid',
                   headers=['Dataset Summary', file]))

    return data_pt


def get_distance(pt1: np.ndarray, pt2: np.ndarray, manhattan=False):
    if manhattan:
        return np.sum(np.abs(pt1 - pt2), axis=-1)
    return np.sqrt(np.sum((pt1 - pt2) ** 2, axis=-1))


def init_meloid(data_pt: np.ndarray, k: int):
    centroid_idx = np.random.randint(low=0, high=data_pt.shape[0] + 1, size=k)

    return centroid_idx


def get_cluster_assignment_with_cost(data: np.ndarray, k: int, medoid_idx):
    medoids = data[medoid_idx]
    # print(medoids)
    dist = np.array([get_distance(data, m) for m in medoids])
    cluster_assignment = np.argmin(dist, axis=0)
    # print(dist.shape)
    # print(cluster_assignment.shape)

    min_dist = dist[cluster_assignment,np.arange(dist.shape[1])]
    cost = np.sum(min_dist)
    # print(cost)
    return cluster_assignment, cost


def k_medoids(data: np.ndarray, k: int, max_iter=20, visualize=False):
    n_sample, n_feat = data.shape
    medoid_idx = init_meloid(data_pt, k)

    if visualize:
        viz = Cluster_viz(data)

    cluster_assignment, old_cost = get_cluster_assignment_with_cost(data, k, medoid_idx)
    print('init---', 'medoids', medoid_idx, 'cost', old_cost)
    if visualize:
        viz.visualize_iteration(0 , cluster_assignment)

    for _it in range(max_iter):
        swap_flag = False
        for n in range(n_sample):
            if n in medoid_idx:
                continue
            for swap_pos in range(k):
                new_medoid_idx = deepcopy(medoid_idx)
                new_medoid_idx[swap_pos] = n

                _new_cluster_assignment, new_cost = get_cluster_assignment_with_cost(data, k, new_medoid_idx)

                if new_cost < old_cost:
                    # print('swapped', medoid_idx[swap_pos], 'with', n)
                    swap_flag = True
                    medoid_idx = new_medoid_idx
                    old_cost = new_cost
                    cluster_assignment = _new_cluster_assignment
        if visualize:
            viz.visualize_iteration(_it+1, cluster_assignment)
        if swap_flag is False:
            print('medoids', data[medoid_idx])
            # print(cluster_assignment)
            cl, member_count =  np.unique(cluster_assignment, return_counts=True)
            table= [[cl[i],member_count[i]] for i in range(len(cl))]
            print(tabulate(table, headers=['Cluster', '# of Members'], tablefmt="fancy_grid"))
            print('end')
            break
        print('iteration', _it+1, 'medoids', medoid_idx, 'cost', old_cost)


if __name__ == '__main__':
    data_pt = load_dataset('Dataset/buddymove_holidayiq.csv', exclude_cols=[0])

    # print(data_pt[0:5,:]-data_pt[1,:])
    # print(get_distance(data_pt[0:5,:],data_pt[1,:]))
    # print(get_distance(data_pt[0,:],data_pt[1,:],manhattan=False))

    k_medoids(data_pt, 4, visualize=True)
