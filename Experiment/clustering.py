# from __future__ import print_function
# import time
import numpy as np
import pandas as pd
# matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import ipyvolume as ipv
from sklearn.cluster import SpectralClustering, DBSCAN, OPTICS, AgglomerativeClustering
from plot import create_bar_plot, create_2d_scatterplot, create_3d_scatterplot, create_2d_scatterplot_clustering, create_clustering_plot, create_2d_pyplot, create_reachability_plot
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from clusterEvaluation import cluster_evaluation


def spectral_clustering(df, dataType="", graphs=False):
    print("\n- spectral clustering (" + dataType + ")...")
    sc = SpectralClustering(
        n_clusters=3, assign_labels="discretize", random_state=0)
    clustering = sc.fit(df)
    if graphs == True:
        create_clustering_plot(
            sc, df, "Spectral Clustering (" + dataType + ")")
    cluster_evaluation(sc, df)


def dbscan_clustering(df, dataType="", graphs=False):
    print("\n- dbscan clustering (" + dataType + ")...")
    db = DBSCAN(eps=2, min_samples=4)
    # db = DBSCAN(eps=3, min_samples=2) - with     tsne = TSNE(n_components=number_components,init='random', perplexity=50, n_iter=5000, learning_rate=200)
    clustering = db.fit(df)
    # print(db)
    # print(str(len(df.columns)))
    if graphs == True:
        create_clustering_plot(db, df, "DBSCAN (" + dataType + ")")
        # create_3d_scatterplot_clustering
    return cluster_evaluation(db, df)


def predict_eps_dbscan_parameter(df):
    print("predict Eps DBSCAN parameter...")
    nbrs = NearestNeighbors(
        n_neighbors=4, algorithm='ball_tree', metric='euclidean')
    nbrsResult = nbrs.fit(df)
    distance, ind = nbrs.kneighbors(df)
    # We have no use of indices here
    # dist is a 2 dimensional array with the rows and distances (each row is list of length 4 - distances to 4 nearest neighbours).
    # get distance to only the 4th nearest neighbour
    distances = [distance[i][3] for i in range(len(distance))]
    distances.sort(reverse=True)
    create_2d_pyplot(distances)
    # plt.plot(distances)
    # plt.ylabel('n-dist sorted graph')
    # plt.show()
    # print(distances)

    # print(nbrsResult)
    # A = nbrsResult.kneighbors_graph(df)
    # A.toarray()
    # print(A)
#
    # knn = KNeighborsClassifier(n_neighbors=4, metric='euclidean')
    # knnResult = knn.fit(df)
    # print(knn)


def optics_clustering(df, dataType="", graphs=False):
    print("\n- optics clustering (" + dataType + ")...")
    optics = OPTICS(min_samples=10,  min_cluster_size=0.05)
    # optics = OPTICS(cluster_method='dbscan', eps=2)
    # optics = OPTICS(min_samples=4, cluster_method="xi")
    clustering = optics.fit(df)
    if graphs == True:
        create_clustering_plot(
            optics, df, "OPTICS (" + dataType + ")")

    create_reachability_plot(df, clustering)

    # space = np.arange(len(df))
    # reachability = clustering.reachability_[clustering.ordering_]
    # labels = clustering.labels_[clustering.ordering_]

    # # Reachability plot
    # colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    # for klass, color in zip(range(0, 5), colors):
    #     Xk = space[labels == klass]
    #     Rk = reachability[labels == klass]
    #     plt.plot(Xk, Rk, color, alpha=0.3)
    # plt.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    # plt.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
    # plt.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
    # # plt.set_ylabel('Reachability (epsilon distance)')
    # # plt.set_title('Reachability Plot')
    # plt.show()

    # # Defining the framework of the visualization
    # plt.figure(figsize=(10, 7))
    # # G = gridspec.GridSpec(2, 3)
    # ax1 = plt.subplot()

    # # Plotting the Reachability-Distance Plot
    # colors = ['c.', 'b.', 'r.', 'y.', 'g.']
    # for Class, colour in zip(range(0, 5), colors):
    #     Xk = space[labels == Class]
    #     Rk = reachability[labels == Class]
    #     ax1.plot(Xk, Rk, colour, alpha=0.3)
    # ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    # ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
    # ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
    # ax1.set_ylabel('Reachability Distance')
    # ax1.set_title('Reachability Plot')
    # plt.show()
    # return cluster_evaluation(optics, df)


def agglomerative_clustering(df, dataType="", graphs=False):
    print("\n- agglomerative clustering (" + dataType + ")...")
    agglomerative = AgglomerativeClustering()
    clustering = agglomerative.fit(df)
    if graphs == True:
        create_clustering_plot(
            agglomerative, df, "Agglomerative Clustering (" + dataType + ")")
    return cluster_evaluation(agglomerative, df)
