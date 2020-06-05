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


def spectral_clustering(df):
    print("spectral clustering...")
    print(df)
    sc = SpectralClustering(
        n_clusters=3, assign_labels="discretize", random_state=0)
    clustering = sc.fit(df)
    create_2d_scatterplot_clustering(sc, df)


def dbscan_clustering(df):
    print("dbscan clustering...")
    db = DBSCAN(eps=3, min_samples=2)
    clustering = db.fit(df)
    create_2d_scatterplot_clustering(db, df)


def optics_clustering(df):
    print("optics clustering...")
    optics = OPTICS(min_samples=2)
    clustering = optics.fit(df)
    create_2d_scatterplot_clustering(optics, df)


def agglomerative_clustering(df):
    print("agglomerative clustering...")
    agglomerative = AgglomerativeClustering()
    clustering = agglomerative.fit(df)
    create_2d_scatterplot_clustering(agglomerative, df)


# -----ipyvolume scatterplot---

# x = df["pca-one"],
# y = df["pca-two"],
# z = df["pca-three"],

# fig = ipv.figure()
# scatter = ipv.scatter(x, y, z)
# ipv.show()
