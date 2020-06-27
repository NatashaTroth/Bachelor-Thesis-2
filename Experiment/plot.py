import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import seaborn as sns
import random


def create_bar_plot(x, y, xName, yName):
    plt.figure(figsize=(20, 10))
    data = pd.DataFrame({xName: x, yName: y})
    sns.barplot(x=xName, y=yName, data=data)

    plt.show()


def create_2d_scatterplot(data, xName, yName, title):
    plt.figure(figsize=(20, 10))
    p1 = sns.scatterplot(
        x=xName, y=yName,
        data=data,
        legend="full",
        alpha=0.3,

    )
    df = pd.DataFrame(data[[xName, yName]])

    # add indices to data points in the scatter plot
    # for line in range(0, data.shape[0]):
    #     p1.text(df.iat[line, 0]+0.2, df.iat[line, 1], line,
    #             horizontalalignment='left', size=4, color='black', weight='regular')
    plt.title(title)
    plt.show()


def create_2d_scatterplot_tester_colors(data, xName, yName, colors, title):
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
    ax.scatter(
        xs=data[xName],
        ys=data[yName],
        zs=data[zName],
        c=colors
    )
    plt.title(title)
    ax.set_xlabel(xName)
    ax.set_ylabel(yName)
    ax.set_zlabel(zName)
    plt.show()


def create_clustering_plot(clustering_method, df, title):
    if len(df.columns) == 2:
        create_2d_scatterplot_clustering(clustering_method, df, title)
    if len(df.columns) == 3:
        create_3d_scatterplot_clustering(clustering_method, df, title)


def create_2d_scatterplot_clustering(clustering_method, df, title):
    plt.figure(figsize=(20, 10))
    cluster_labels = clustering_method.fit_predict(df)
    colors = get_array_random_colors(len(np.unique(cluster_labels)) - 1)
    cluster_colors = []
    i = 0
    while i < len(cluster_labels):
        cluster_colors.append(colors[cluster_labels[i]])
        i += 1

    print("PLOT cluster LABELS")
    print(cluster_labels)
    plt.scatter(df[0], df[1], c=cluster_colors, cmap='Paired')
    # plt.scatter(df[0], df[1], c=cluster_labels, cmap='Paired')
    plt.title(title)
    plt.show()


def create_3d_scatterplot_clustering(clustering_method, df, title):
    ax = plt.figure(figsize=(20, 10)).gca(projection='3d')
    cluster_labels = clustering_method.fit_predict(df)
    ax.scatter(
        xs=df[0],
        ys=df[1],
        zs=df[2],
        c=cluster_labels
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title(title)
    plt.show()


def create_2d_pyplot(data):
    plt.figure(figsize=(20, 10))
    plt.plot(data, "ro")
    plt.title('n-dist sorted graph')
    plt.ylabel('distance to 4th nearest neighbor')
    plt.show()


def get_array_random_colors(size):
    random.seed(42)
    colors = []
    i = 0
    while i < size:

        random_color = "#" + "%06x" % random.randint(0, 0xFFFFFF)
        colors.append(random_color)
        i += 1
    return colors


def create_reachability_plot(df, clustering):

    print("COLORS")
    print(matplotlib.colors)
    space = np.arange(len(df))
    reachability = clustering.reachability_[clustering.ordering_]
    labels = clustering.labels_[clustering.ordering_]
    # Defining the framework of the visualization
    plt.figure(figsize=(50, 10))
    # # G = gridspec.GridSpec(2, 3)
    # plt.plot(reachability)
    # plt.ylabel('some numbers')
    # plt.show()

    print("LABELS")
    print(np.unique(labels))
    unique_labels = np.unique(labels)

    ax1 = plt.subplot()
    # ax1 = freq_series.plot(kind='bar')

    # Plotting the Reachability-Distance Plot

    colors = get_array_random_colors(len(unique_labels) - 1)
    for Class, color in zip(range(0, unique_labels[len(unique_labels) - 1]), colors):
        print(Class)
        # for Class, colour in zip(range(0, 5), colors):
        # for all points with labels (clusters), first cluster 1, then 2...
        Xk = space[labels == Class]
        # print("Xk")
        # print(Xk)
        Rk = reachability[labels == Class]
        # print("Rk")
        # print(Rk)
        ax1.bar(Xk, Rk, color=color, width=1)
    ax1.bar(space[labels == -1], reachability[labels == -1],
            color='k', width=1)
    ax1.plot(space, np.full_like(space, 2., dtype=float), color='k', alpha=0.5)
    ax1.plot(space, np.full_like(space, 0.5, dtype=float), color='k', alpha=0.5)
    ax1.fill_between(0, space)
    ax1.set_ylim([0, 7.5])
    ax1.set_ylabel('Reachability Distance')
    ax1.set_title('Reachability Plot')
    plt.show()
