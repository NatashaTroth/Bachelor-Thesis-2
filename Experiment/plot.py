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
    plt.figure(figsize=(20, 10))
    data = pd.DataFrame({xName: x, yName: y})
    sns.barplot(x=xName, y=yName, data=data)

    plt.show()


def create_2d_scatterplot(data, xName, yName, title):
    plt.figure(figsize=(20, 10))
    p1 = sns.scatterplot(
        x=xName, y=yName,
        # x="pca-one", y="pca-two",
        # hue="pca-one",
        # palette=sns.color_palette("hls", 1),
        data=data,
        legend="full",
        alpha=0.3,

    )
    # rows, cols = df.shape
    # add annotations one by one with a loop
    df = pd.DataFrame(data[[xName, yName]])

    print(df)
    print("---------------------")
    print(df.iat[0, 0])
    print(df.iat[0, 1])
    print(df.iat[1, 1])
    # for line in range(0, data.shape[0]):
    #     p1.text(df.iat[line, 0]+0.2, df.iat[line, 1], line,
    #             horizontalalignment='left', size=4, color='black', weight='regular')
    plt.title(title)
    plt.show()


def create_2d_scatterplot_tester_colors(data, xName, yName, colors, title):
    print("CREATING TSNE PLOT-------")
    df = pd.DataFrame(data[[xName, yName]])
    plt.figure(figsize=(20, 10))

    # for i in range(0, data.shape[0]):
    #     plt.scatter(df.iat[0, i], df.iat[1, i], c=colors[i])
    p1 = plt.scatter(df[xName], df[yName])
    # p1 = plt.scatter(df[xName], df[yName], c=colors)
    # for line in range(0, data.shape[0]):
    #     plt.text(df.iat[line, 0], df.iat[line, 1], line,
    #              horizontalalignment='left', size=4, color='black', weight='regular')
    plt.title(title)
    plt.show()


def create_3d_scatterplot(data, xName, yName, zName, colors, title):
    ax = plt.figure(figsize=(20, 10)).gca(projection='3d')
    # label_color_dict = {'pca-one': 'magenta', 'pca-two': 'orange',
    #                     'pca-three': 'blue'}
    # labels = ["pca-one", "pca-two", "pca-three"]
    # cvec = [label_color_dict[label] for label in labels]
    ax.scatter(
        xs=data[xName],
        ys=data[yName],
        zs=data[zName],
        c=colors
        # c=list(range(0, 8)),
        # cmap='tab10'
    )
    plt.title(title)
    ax.set_xlabel(xName)
    ax.set_ylabel(yName)
    ax.set_zlabel(zName)
    plt.show()


def create_clustering_plot(clustering_method, df, title):
    # print(str(len(df.columns)))
    if len(df.columns) == 2:
        create_2d_scatterplot_clustering(clustering_method, df, title)
    if len(df.columns) == 3:
        create_3d_scatterplot_clustering(clustering_method, df, title)


def create_2d_scatterplot_clustering(clustering_method, df, title):
    plt.figure(figsize=(20, 10))
    cluster_labels = clustering_method.fit_predict(df)
    plt.scatter(df[0], df[1], c=cluster_labels, cmap='Paired')
    plt.title(title)
    plt.show()


# def create_2d_scatterplot_clustering(clustering_method, df, title):
#     # plt.figure(figsize=(10, 7))
#     cluster_labels = clustering_method.fit_predict(df)

#     rows, cols = df.shape
#     fig, ax = plt.subplots(figsize=(50/10, 25/10))
#     # df.plot(ax=ax)
#     ax.scatter(df[0], df[1], c=cluster_labels, cmap='Paired')
#     plt.title(title)
#     # print(list(df.index.values))

#     for col in range(cols):
#         for i in range(rows):
#             ax.annotate('{}'.format(df.iloc[i, col]), xy=(i, df.iloc[i, col]))

#     # for i in list(df.index.values):
#     #     plt.text(df[i, 0], df[i, 1], str(i))
#     # for i in list(df.index.values):
#     #     plt.text(df[i, 0], df[i, 1], str(i))
#     plt.show()


def create_3d_scatterplot_clustering(clustering_method, df, title):

    ax = plt.figure(figsize=(20, 10)).gca(projection='3d')
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

    # plt.figure(figsize=(20,10)).gca(projection='3d')
    # cluster_labels = clustering_method.fit_predict(df)
    # plt.scatter(xs=df[0], ys=df[1], zs=df[2], c=cluster_labels)
    plt.title(title)
    plt.show()


def create_2d_pyplot(data):
    plt.figure(figsize=(20, 10))
    plt.plot(data, "ro")
    plt.title('n-dist sorted graph')
    plt.ylabel('distance to 4th nearest neighbor')
    plt.show()


def create_reachability_plot(df, clustering):

    space = np.arange(len(df))
    reachability = clustering.reachability_[clustering.ordering_]
    labels = clustering.labels_[clustering.ordering_]
    # Defining the framework of the visualization
    plt.figure(figsize=(50, 10))
    # G = gridspec.GridSpec(2, 3)
    ax1 = plt.subplot()

    # Plotting the Reachability-Distance Plot
    colors = ['c.', 'b.', 'r.', 'y.', 'g.']
    for Class, colour in zip(range(0, 5), colors):
        Xk = space[labels == Class]
        Rk = reachability[labels == Class]
        ax1.plot(Xk, Rk, colour, alpha=0.3)
    ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
    ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
    ax1.set_ylabel('Reachability Distance')
    ax1.set_title('Reachability Plot')
    plt.show()

# -----ipyvolume scatterplot---

# x = df["pca-one"],
# y = df["pca-two"],
# z = df["pca-three"],

# fig = ipv.figure()
# scatter = ipv.scatter(x, y, z)
# ipv.show()
