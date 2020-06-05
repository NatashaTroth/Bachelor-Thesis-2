# from __future__ import print_function
# import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import ipyvolume as ipv
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN


def create_bar_plot(x, y, xName, yName):
    plt.figure(figsize=(16, 10))
    data = pd.DataFrame({xName: x, yName: y})
    sns.barplot(x=xName, y=yName, data=data)

    plt.show()


def create_2DScatterplot(data, xName, yName):
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x=xName, y=yName,
        # x="pca-one", y="pca-two",
        # hue="pca-one",
        # palette=sns.color_palette("hls", 1),
        data=data,
        legend="full",
        alpha=0.3
    )
    plt.show()


def create_3DScatterplot(data, xName, yName, zName):
    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    # label_color_dict = {'pca-one': 'magenta', 'pca-two': 'orange',
    #                     'pca-three': 'blue'}
    # labels = ["pca-one", "pca-two", "pca-three"]
    # cvec = [label_color_dict[label] for label in labels]
    ax.scatter(
        xs=data[xName],
        ys=data[yName],
        zs=data[zName],
        # c=cvec
        # c=list(range(0, 8)),
        # cmap='tab10'
    )
    ax.set_xlabel(xName)
    ax.set_ylabel(yName)
    ax.set_zlabel(zName)
    plt.show()
