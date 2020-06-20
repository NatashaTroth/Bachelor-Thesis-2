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
    clustering = db.fit(df)
    cluster_labels = db.fit_predict(df)
    if graphs == True:
        create_clustering_plot(db, df, "DBSCAN (" + dataType + ")")
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


def optics_clustering(df, dataType="", graphs=False):
    print("\n- optics clustering (" + dataType + ")...")
    # optics = OPTICS(min_samples=10,  min_cluster_size=0.05)
    # optics = OPTICS(cluster_method='dbscan', eps=2)
    optics = OPTICS(min_samples=4, cluster_method="xi")
    clustering = optics.fit(df)
    if graphs == True:
        create_clustering_plot(
            optics, df, "OPTICS (" + dataType + ")")
    create_reachability_plot(df, clustering)
    return cluster_evaluation(optics, df)


def agglomerative_clustering(df, dataType="", graphs=False):
    print("\n- agglomerative clustering (" + dataType + ")...")
    agglomerative = AgglomerativeClustering()
    clustering = agglomerative.fit(df)
    if graphs == True:
        create_clustering_plot(
            agglomerative, df, "Agglomerative Clustering (" + dataType + ")")
    return cluster_evaluation(agglomerative, df)
