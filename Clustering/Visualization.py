import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


class Cluster_viz:
    def __init__(self, data: np.ndarray):
        pca = PCA(n_components=2).fit(data)
        self.pca_data = pca.transform(data)
        # print(self.pca_data.shape)

    def visualize_iteration(self, iteration, cluster_assignment):
        plt.scatter(self.pca_data[:, 0], self.pca_data[:, 1], c=cluster_assignment)
        plt.title('Iteration ' + str(iteration))
        plt.show(block=False)
        plt.pause(1)
        # plt.close()
