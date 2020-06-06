# from __future__ import print_function
# import time
import numpy as np
import pandas as pd
# matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import ipyvolume as ipv
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from plot import create_bar_plot
from plot import create_2d_scatterplot
from plot import create_3d_scatterplot
from plot import create_2d_scatterplot_clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def spectral_clustering(df, dataType="", graphs=False):
    print("\n- spectral clustering (" + dataType + ")...")
    sc = SpectralClustering(
        n_clusters=3, assign_labels="discretize", random_state=0)
    clustering = sc.fit(df)
    if graphs == True:
        create_2d_scatterplot_clustering(
            sc, df, "Spectral Clustering (" + dataType + ")")
    cluster_evaluation(sc, df)


def dbscan_clustering(df, dataType="", graphs=False):
    print("\n- dbscan clustering (" + dataType + ")...")
    db = DBSCAN(eps=3, min_samples=2)
    clustering = db.fit(df)
    if graphs == True:
        create_2d_scatterplot_clustering(db, df, "DBSCAN (" + dataType + ")")
    cluster_evaluation(db, df)


def optics_clustering(df, dataType="", graphs=False):
    print("\n- optics clustering (" + dataType + ")...")
    optics = OPTICS(min_samples=2)
    clustering = optics.fit(df)
    if graphs == True:
        create_2d_scatterplot_clustering(
            optics, df, "OPTICS (" + dataType + ")")
    cluster_evaluation(optics, df)


def agglomerative_clustering(df, dataType="", graphs=False):
    print("\n- agglomerative clustering (" + dataType + ")...")
    agglomerative = AgglomerativeClustering()
    clustering = agglomerative.fit(df)
    if graphs == True:
        create_2d_scatterplot_clustering(
            agglomerative, df, "Agglomerative Clustering (" + dataType + ")")
    cluster_evaluation(agglomerative, df)


# -------- Evaluation Methods --------

def cluster_evaluation(clustering_method, df):
    silhouette_score_evaluation(clustering_method, df)
    davies_bouldin_score_evaluation(clustering_method, df)
    calinski_harabasz_score_evaluation(clustering_method, df)


def silhouette_score_evaluation(clustering_method, df):
    # print("calculation silhouette score...")
    cluster_labels = clustering_method.fit_predict(df)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    if len(np.unique(cluster_labels)) > 1:
        score = silhouette_score(df, cluster_labels)
        print("Silhouette score: " + str(score))
    else:
        print("Only one cluster - there have to be at least 2 clusters to calculation the Silhouette score.")


def davies_bouldin_score_evaluation(clustering_method, df):
    # print("calculation davies bouldin score...")

    cluster_labels = clustering_method.fit_predict(df)

    if len(np.unique(cluster_labels)) > 1:
        score = davies_bouldin_score(df, cluster_labels)
        print("Davies Bouldin score: " + str(score))
    else:
        print("Only one cluster - there have to be at least 2 clusters to calculation the Davies Bouldin score.")


def calinski_harabasz_score_evaluation(clustering_method, df):
    # print("calculation calinski harabasz score...")
    cluster_labels = clustering_method.fit_predict(df)
    if len(np.unique(cluster_labels)) > 1:
        score = calinski_harabasz_score(df, cluster_labels)
        print("Calinski harabasz score: " + str(score))
    else:
        print("Only one cluster - there have to be at least 2 clusters to calculation the Calinski harabasz score.")
