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


def calculate_PCA(df):
    print("calculating PCA...")
    pca = PCA()
    pca_result = pca.fit_transform(df[df.columns].values)
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]
    df['pca-three'] = pca_result[:, 2]
    print('Explained variation per principal component: {}'.format(
        pca.explained_variance_ratio_))
    print('Explained variation per principal component cumulative: {}'.format(
        pca.explained_variance_ratio_.cumsum()))
    # createBarPlot(list(range(0, len(pca.explained_variance_ratio_))), pca.explained_variance_ratio_,
    #               'Principle Components (ordered by highest variance to lowest)', 'Variance Ratio')

    # print(df)
    create_2DScatterplot(df)
    create_3DScatterplot(df)

    # sns.barplot(x=test, y=pca.explained_variance_ratio_)
    # plt.show()
    # pca = PCA(n_components=3)
    # pca_result = pca.fit_transform(df[df.columns].values)
    # df['pca-one'] = pca_result[:, 0]
    # df['pca-two'] = pca_result[:, 1]
    # df['pca-three'] = pca_result[:, 2]
    # print('Explained variation per principal component: {}'.format(
    #     pca.explained_variance_ratio_))
    # print(pca_result[:, 0])


def create_bar_plot(x, y, xName, yName):
    plt.figure(figsize=(16, 10))
    data = pd.DataFrame({xName: x, yName: y})
    sns.barplot(x=xName, y=yName, data=data)

    plt.show()


def create_2DScatterplot(data):
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        # hue="pca-one",
        # palette=sns.color_palette("hls", 1),
        data=data,
        legend="full",
        alpha=0.3
    )
    plt.show()


def create_3DScatterplot(data):
    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    # label_color_dict = {'pca-one': 'magenta', 'pca-two': 'orange',
    #                     'pca-three': 'blue'}
    # labels = ["pca-one", "pca-two", "pca-three"]
    # cvec = [label_color_dict[label] for label in labels]
    ax.scatter(
        xs=data["pca-one"],
        ys=data["pca-two"],
        zs=data["pca-three"],
        # c=cvec
        # c=list(range(0, 8)),
        # cmap='tab10'
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    plt.show()


# -----ipyvolume scatterplot---

# x = df["pca-one"],
# y = df["pca-two"],
# z = df["pca-three"],

# fig = ipv.figure()
# scatter = ipv.scatter(x, y, z)
# ipv.show()
