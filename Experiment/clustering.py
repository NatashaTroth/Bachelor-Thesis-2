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


def calculatePCA(df):
    print("calculating PCA...")
    pca = PCA()
    pca_result = pca.fit_transform(df[df.columns].values)
    # df['pca-one'] = pca_result[:, 0]
    # df['pca-two'] = pca_result[:, 1]
    # df['pca-three'] = pca_result[:, 2]
    print('Explained variation per principal component: {}'.format(
        pca.explained_variance_ratio_))
    print('Explained variation per principal component cumulative: {}'.format(
        pca.explained_variance_ratio_.cumsum()))
    # print(pca_result[:, 0])
    plt.figure(figsize=(16, 10))
    createBarPlot(list(range(0, len(pca.explained_variance_ratio_))), pca.explained_variance_ratio_,
                  'Principle Components (ordered by highest variance to lowest)', 'Variance Ratio')
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


def createBarPlot(x, y, xName, yName):
    data = pd.DataFrame({xName: x, yName: y})
    sns.barplot(x=xName, y=yName, data=data)

    plt.show()

# -----seaborn scatterplot---
    # plt.figure(figsize=(16, 10))
    # sns.scatterplot(
    #     x="pca-one", y="pca-two",
    #     hue="pca-one",
    #     palette=sns.color_palette("hls", 1),
    #     data=df,
    #     legend="full",
    #     alpha=0.3
    # )
    # plt.show()

# -----mathlib 3d scatterplot---
    # ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    # ax.scatter(
    #     xs=df["pca-one"],
    #     ys=df["pca-two"],
    #     zs=df["pca-three"],
    #     # c=df["y"],
    #     # cmap='tab10'
    # )
    # ax.set_xlabel('pca-one')
    # ax.set_ylabel('pca-two')
    # ax.set_zlabel('pca-three')
    # plt.show()


# -----ipyvolume scatterplot---

    # x = df["pca-one"],
    # y = df["pca-two"],
    # z = df["pca-three"],

    # fig = ipv.figure()
    # scatter = ipv.scatter(x, y, z)
    # ipv.show()
