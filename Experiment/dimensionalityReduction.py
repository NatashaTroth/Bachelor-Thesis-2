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
from sklearn.cluster import SpectralClustering, DBSCAN
from plot import create_bar_plot, create_2d_scatterplot, create_3d_scatterplot


def calculate_PCA(df, number_components, graphs):
    print("calculating PCA...")
    pca = PCA(n_components=number_components)
    pca_results = pca.fit_transform(df[df.columns].values)
    df['pca-one'] = pca_results[:, 0]
    df['pca-two'] = pca_results[:, 1]

    if number_components > 2:
        df['pca-three'] = pca_results[:, 2]

    print('Explained variation per principal component: {}'.format(
        pca.explained_variance_ratio_))
    print('Explained variation per principal component cumulative: {}'.format(
        pca.explained_variance_ratio_.cumsum()))

    if graphs == True:
        # print(len(pca_result[0]))
        create_bar_plot(list(range(0, len(pca.explained_variance_ratio_))), pca.explained_variance_ratio_,
                        'Principle Components (ordered by highest variance to lowest)', 'Variance Ratio')
        if number_components == 2:
            create_2d_scatterplot(df, "pca-one", "pca-two")
        if number_components == 3:
            create_3d_scatterplot(df, "pca-one", "pca-two", "pca-three")

    # calculate_TSNE(pd.DataFrame(pca_result[:, [0, 3]]))
    # print(pca_results)
    return pd.DataFrame(pca_results)


def calculate_TSNE(df, number_components, graphs):
    print("calculating TSNE...")
    tsne = TSNE(n_components=number_components,
                init='random', perplexity=50, n_iter=5000, learning_rate=50)

    # very good
    # tsne = TSNE(n_components=number_components,
    #             init='random', perplexity=50, n_iter=5000, learning_rate=10)
    # tsne = TSNE(n_components=number_components,
    #             init='random', perplexity=25, n_iter=5000, learning_rate=100)

    # Goodish:
    # tsne = TSNE(n_components=number_components,
    #             init='random', perplexity=25, n_iter=2000, learning_rate=10)
    # tsne = TSNE(n_components=2, verbose=0,
    #             perplexity=40, n_iter=300, init='random')
    tsne_results = tsne.fit_transform(df)
    # print(df)
    # print(tsne_results)

    df['tsne-one'] = tsne_results[:, 0]
    df['tsne-two'] = tsne_results[:, 1]

    if number_components > 2:
        df['tsne-three'] = tsne_results[:, 2]

    if graphs == True:
        if number_components == 2:
            create_2d_scatterplot(df, "tsne-one", "tsne-two")
        if number_components == 3:
            create_3d_scatterplot(df, "tsne-one", "tsne-two", "tsne-three")

    return pd.DataFrame(tsne_results)
