from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


class Cluster_viz:
    def __init__(self, data:np.ndarray):
        pca = PCA(n_components=2).fit(data)
        self.pca_data = pca.transform(data)
        print(self.pca_data.shape)


    def visualize_iteration(self, iteration,cluster_assignment):
        plt.scatter(self.pca_data[:,0], self.pca_data[:,1], c=cluster_assignment)
        # plt.title('Iteration', iteration)
        plt.show()


