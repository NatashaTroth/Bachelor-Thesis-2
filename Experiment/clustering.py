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
from plot import create_bar_plot, create_2d_scatterplot, create_3d_scatterplot, create_2d_scatterplot_clustering, create_clustering_plot
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
    db = DBSCAN(eps=3, min_samples=2)
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
    print(nbrsResult)
    A = nbrsResult.kneighbors_graph(df)
    A.toarray()
    print(A)
#
    # knn = KNeighborsClassifier(n_neighbors=4, metric='euclidean')
    # knnResult = knn.fit(df)
    # print(knn)


def optics_clustering(df, dataType="", graphs=False):
    print("\n- optics clustering (" + dataType + ")...")
    optics = OPTICS(min_samples=2, cluster_method="xi")
    clustering = optics.fit(df)
    if graphs == True:
        create_clustering_plot(
            optics, df, "OPTICS (" + dataType + ")")
    return cluster_evaluation(optics, df)


def agglomerative_clustering(df, dataType="", graphs=False):
    print("\n- agglomerative clustering (" + dataType + ")...")
    agglomerative = AgglomerativeClustering()
    clustering = agglomerative.fit(df)
    if graphs == True:
        create_clustering_plot(
            agglomerative, df, "Agglomerative Clustering (" + dataType + ")")
    return cluster_evaluation(agglomerative, df)


# # -------- Evaluation Methods --------

# def cluster_evaluation(clustering_method, df):
#     cluster_evaluation_scores = []
#     cluster_evaluation_scores.append(silhouette_score_evaluation(
#         clustering_method, df))
#     cluster_evaluation_scores.append(davies_bouldin_score_evaluation(
#         clustering_method, df))
#     cluster_evaluation_scores.append(calinski_harabasz_score_evaluation(
#         clustering_method, df))
#     return cluster_evaluation_scores


# def silhouette_score_evaluation(clustering_method, df):
#     # print("calculation silhouette score...")
#     cluster_labels = clustering_method.fit_predict(df)
#     print("labels: ")
#     print(np.unique(cluster_labels))
#     # print(cluster_labels)
#     # The silhouette_score gives the average value for all the samples.
#     # This gives a perspective into the density and separation of the formed
#     # clusters
#     if len(np.unique(cluster_labels)) > 1:
#         score = silhouette_score(df, cluster_labels)
#         print("Silhouette score: " + str(score))
#         return score
#     else:
#         print("Only one cluster - there have to be at least 2 clusters to calculation the Silhouette score.")
#         return None


# def davies_bouldin_score_evaluation(clustering_method, df):
#     # print("calculation davies bouldin score...")

#     cluster_labels = clustering_method.fit_predict(df)

#     if len(np.unique(cluster_labels)) > 1:
#         score = davies_bouldin_score(df, cluster_labels)
#         print("Davies Bouldin score: " + str(score))
#         return score
#     else:
#         print("Only one cluster - there have to be at least 2 clusters to calculation the Davies Bouldin score.")
#         return None


# def calinski_harabasz_score_evaluation(clustering_method, df):
#     # print("calculation calinski harabasz score...")
#     cluster_labels = clustering_method.fit_predict(df)
#     if len(np.unique(cluster_labels)) > 1:
#         score = calinski_harabasz_score(df, cluster_labels)
#         print("Calinski Harabasz score: " + str(score))
#         return score
#     else:
#         print("Only one cluster - there have to be at least 2 clusters to calculation the Calinski Harabasz score.")
#         return None
