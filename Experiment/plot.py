# from __future__ import print_function
# import time
import numpy as np
import pandas as pd
# matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import ipyvolume as ipv


def create_bar_plot(x, y, xName, yName):
    plt.figure(figsize=(16, 10))
    data = pd.DataFrame({xName: x, yName: y})
    sns.barplot(x=xName, y=yName, data=data)

    plt.show()


def create_2d_scatterplot(data, xName, yName):
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


def create_3d_scatterplot(data, xName, yName, zName):
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


def create_clustering_plot(clustering_method, df, title):
    print(str(len(df.columns)))
    if len(df.columns) == 2:
        create_2d_scatterplot_clustering(clustering_method, df, title)
    if len(df.columns) == 3:
        create_3d_scatterplot_clustering(clustering_method, df, title)


def create_2d_scatterplot_clustering(clustering_method, df, title):
    plt.figure(figsize=(10, 7))
    cluster_labels = clustering_method.fit_predict(df)
    plt.scatter(df[0], df[1], c=cluster_labels, cmap='Paired')
    plt.title(title)
    plt.show()


def create_3d_scatterplot_clustering(clustering_method, df, title):

    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    cluster_labels = clustering_method.fit_predict(df)

    # label_color_dict = {'pca-one': 'magenta', 'pca-two': 'orange',
    #                     'pca-three': 'blue'}
    # labels = ["pca-one", "pca-two", "pca-three"]
    # cvec = [label_color_dict[label] for label in labels]
    ax.scatter(
        xs=df[0],
        ys=df[1],
        zs=df[2],
        # c=cvec
        # c=list(range(0, 8)),
        # cmap='tab10'
        c=cluster_labels
    )
    ax.set_xlabel("test1")
    ax.set_ylabel("test2")
    ax.set_zlabel("test3")

    # plt.figure(figsize=(16, 10)).gca(projection='3d')
    # cluster_labels = clustering_method.fit_predict(df)
    # plt.scatter(xs=df[0], ys=df[1], zs=df[2], c=cluster_labels)
    plt.title(title)
    plt.show()


# -----ipyvolume scatterplot---

# x = df["pca-one"],
# y = df["pca-two"],
# z = df["pca-three"],

# fig = ipv.figure()
# scatter = ipv.scatter(x, y, z)
# ipv.show()
