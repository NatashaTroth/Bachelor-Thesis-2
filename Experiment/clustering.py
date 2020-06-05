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
from plot import create_bar_plot
from plot import create_2DScatterplot
from plot import create_3DScatterplot


def spectral_clustering(df):
    print("spectral clustering...")
    print(df)
    clustering = SpectralClustering(
        n_clusters=3, assign_labels="discretize", random_state=0).fit(df)
    labels_rbf = clustering.fit_predict(df)
    print(labels_rbf)
    # Building the label to colour mapping
    colours = {}
    colours[0] = 'orange'
    colours[1] = 'turquoise'
    colours[2] = 'pink'

    # Building the colour vector for each data point
    cvec = [colours[label] for label in labels_rbf]

    # Plotting the clustered scatter plot

    # b = plt.scatter(df[0], df[1], color='b')
    # y = plt.scatter(df[0], df[1], color='y')

    plt.figure(figsize=(9, 9))
    plt.scatter(df[0], df[1], c=cvec)
    # plt.legend((b, y), ('Label 0', 'Label 1'))
    plt.show()


def dbscan_clustering(df):
    print("dbscan clustering...")
    db = DBSCAN(eps=3, min_samples=2)
    clustering = db.fit(df)

    plt.figure(figsize=(16, 10))
    y_pred = db.fit_predict(df)
    plt.scatter(df[0], df[1], c=y_pred, cmap='Paired')
    plt.title("DBSCAN")

    # print(clustering.labels_)
    # labels = set(clustering.labels_)
    # colours = {}
    # colours[0] = 'orange'
    # colours[1] = 'turquoise'
    # colours[2] = 'pink'
    # colours[3] = 'blue'
    # colours[4] = 'green'
    # colours[5] = 'red'
    # colours[6] = 'purple'

    # # Building the colour vector for each data point
    # cvec = [colours[label] for label in labels]

    # # Plotting the clustered scatter plot

    # # b = plt.scatter(df[0], df[1], color='b')
    # # y = plt.scatter(df[0], df[1], color='y')

    # plt.figure(figsize=(16, 10))
    # plt.scatter(df[0], df[1], c=cvec)
    # # plt.legend((b, y), ('Label 0', 'Label 1'))
    plt.show()

# -----ipyvolume scatterplot---

# x = df["pca-one"],
# y = df["pca-two"],
# z = df["pca-three"],

# fig = ipv.figure()
# scatter = ipv.scatter(x, y, z)
# ipv.show()
