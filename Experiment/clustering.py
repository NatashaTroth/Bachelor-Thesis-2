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

    # print(len(pca_result[0]))
    create_bar_plot(list(range(0, len(pca.explained_variance_ratio_))), pca.explained_variance_ratio_,
                    'Principle Components (ordered by highest variance to lowest)', 'Variance Ratio')
    create_2DScatterplot(df, "pca-one", "pca-two")
    create_3DScatterplot(df, "pca-one", "pca-two", "pca-three")

    # calculate_TSNE(pd.DataFrame(pca_result[:, [0, 3]]))

    return pd.DataFrame(pca_result)


def calculate_TSNE(df):
    print("calculating TSNE...")
    tsne = TSNE(n_components=3, init='random')
    # tsne = TSNE(n_components=2, verbose=0,
    #             perplexity=40, n_iter=300, init='random')
    tsne_results = tsne.fit_transform(df)
    print(df)
    print(tsne_results)

    df['tsne-one'] = tsne_results[:, 0]
    df['tsne-two'] = tsne_results[:, 1]
    df['tsne-three'] = tsne_results[:, 2]
    create_2DScatterplot(df, "tsne-one", "tsne-two")
    create_3DScatterplot(df, "tsne-one", "tsne-two", "tsne-three")

    return pd.DataFrame(tsne_results)


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


def spectral_clustering():
    clustering = SpectralClustering(
        n_clusters=3, assign_labels="discretize", random_state=0).fit(X)

# -----ipyvolume scatterplot---

# x = df["pca-one"],
# y = df["pca-two"],
# z = df["pca-three"],

# fig = ipv.figure()
# scatter = ipv.scatter(x, y, z)
# ipv.show()
