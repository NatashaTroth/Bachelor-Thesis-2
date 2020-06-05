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
from sklearn.metrics import silhouette_score


def spectral_clustering(df):
    print("spectral clustering...")
    print(df)
    sc = SpectralClustering(
        n_clusters=3, assign_labels="discretize", random_state=0)
    clustering = sc.fit(df)
    create_2d_scatterplot_clustering(sc, df)
    silhouette_score_evaluation(sc, df)


def dbscan_clustering(df):
    print("dbscan clustering...")
    db = DBSCAN(eps=3, min_samples=2)
    clustering = db.fit(df)
    create_2d_scatterplot_clustering(db, df)
    silhouette_score_evaluation(db, df)


def optics_clustering(df):
    print("optics clustering...")
    optics = OPTICS(min_samples=2)
    clustering = optics.fit(df)
    create_2d_scatterplot_clustering(optics, df)
    silhouette_score_evaluation(optics, df)


def agglomerative_clustering(df):
    print("agglomerative clustering...")
    agglomerative = AgglomerativeClustering()
    clustering = agglomerative.fit(df)
    create_2d_scatterplot_clustering(agglomerative, df)
    silhouette_score_evaluation(agglomerative, df)


def silhouette_score_evaluation(clustering_method, df):
    print("calculation silhouette score...")
    # print(df)
    # print("--------")

    cluster_labels = clustering_method.fit_predict(df)
    # print("unique length: " + str(len(np.unique(cluster_labels))))
    # print(cluster_labels)
    # print("length labels: " + str(len(cluster_labels)))
    # print("length df: " + str(len(df)))

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    if len(np.unique(cluster_labels)) > 1:
        score = silhouette_score(df, cluster_labels)

    # labels = clustering_method.fit_predict(df)
    # print(np.unique(labels))
    # score = silhouette_score(df, np.unique(labels))
        print("Silhouette score: " + str(score))
    else:
        print("Only one cluster - there have to be at least 2 clusters to calculation the silhouette score.")
# -----ipyvolume scatterplot---

# x = df["pca-one"],
# y = df["pca-two"],
# z = df["pca-three"],

# fig = ipv.figure()
# scatter = ipv.scatter(x, y, z)
# ipv.show()
