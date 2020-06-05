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
from plot import create_bar_plot
from plot import create_2DScatterplot
from plot import create_3DScatterplot


def calculate_PCA(df, graphs):
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

    if graphs == True:
        # print(len(pca_result[0]))
        create_bar_plot(list(range(0, len(pca.explained_variance_ratio_))), pca.explained_variance_ratio_,
                        'Principle Components (ordered by highest variance to lowest)', 'Variance Ratio')
        create_2DScatterplot(df, "pca-one", "pca-two")
        create_3DScatterplot(df, "pca-one", "pca-two", "pca-three")

    # calculate_TSNE(pd.DataFrame(pca_result[:, [0, 3]]))

    return pd.DataFrame(pca_result)


def calculate_TSNE(df, number_components, graphs):
    print("calculating TSNE...")
    tsne = TSNE(n_components=number_components, init='random')
    # tsne = TSNE(n_components=2, verbose=0,
    #             perplexity=40, n_iter=300, init='random')
    tsne_results = tsne.fit_transform(df)
    print(df)
    print(tsne_results)

    df['tsne-one'] = tsne_results[:, 0]
    df['tsne-two'] = tsne_results[:, 1]

    if number_components > 2:
        df['tsne-three'] = tsne_results[:, 2]

    if graphs == True:
        create_2DScatterplot(df, "tsne-one", "tsne-two")
        create_3DScatterplot(df, "tsne-one", "tsne-two", "tsne-three")

    return pd.DataFrame(tsne_results)
