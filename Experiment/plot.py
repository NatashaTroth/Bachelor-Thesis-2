import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as matplotlib
import seaborn as sns
import random


def create_bar_plot(x, y, xName, yName):
    """ render bar plot

        Parameters
        ----------
        x : list
        values for the x-axis
        y : list
        values for the y-axis
        xName : str
        label name for the x-axis
        yName : str
        label name for the y-axis
    """
    plt.figure(figsize=(20, 10))
    data = pd.DataFrame({xName: x, yName: y})
    sns.barplot(x=xName, y=yName, data=data)

    plt.show()


def create_2d_scatterplot(data, xName, yName, colors, title):
    """ render 2D scatter plot with one color for all points, with options to show index labels for each data point and to render different colors for each data point

        Parameters
        ----------
        data : dataFrame
            data for the scatter plot
        xName : str
            label and dataFrame column name for the x-axis
        yName : str
            label and dataFrame column name for the y-axis
        title : str
            title for the scatter plot
    """
    df = pd.DataFrame(data[[xName, yName]])
    plt.figure(figsize=(20, 10))

    # render with one color
    p1 = plt.scatter(df[xName], df[yName])

    # render with given colors
    # p1 = plt.scatter(df[xName], df[yName], c=colors)

    # render with index labels per data point
    # for line in range(0, data.shape[0]):
    #     plt.text(df.iat[line, 0], df.iat[line, 1], line,
    #              horizontalalignment='left', size=4, color='black', weight='regular')
    plt.title(title)
    plt.show()


def create_3d_scatterplot(data, xName, yName, zName, colors, title):
    """ render 3D scatter plot with given colors for all points

        Parameters
        ----------
        data : dataFrame
            data for the scatter plot
        xName : str
            label and dataFrame column name for the x-axis
        yName : str
            label and dataFrame column name for the y-axis
        title : str
            title for the scatter plot
    """
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
    """ create either 2D or 3D clustering scatter plot from given dataFrame

        Parameters
        ----------
        clustering_method : object
            clustering method (before being fit to the dataset)
        df : dataFrame
            data for the scatter plot
        title : str
            title for the scatter plot
    """
    if len(df.columns) == 2:
        create_2d_scatterplot_clustering(clustering_method, df, title)
    if len(df.columns) == 3:
        create_3d_scatterplot_clustering(clustering_method, df, title)


def create_2d_scatterplot_clustering(clustering_method, df, title):
    """ create 2D clustering scatter plot from given dataFrame

        Parameters
        ----------
        clustering_method : object
            clustering method (before being fit to the dataset)
        df : dataFrame
            data for the scatter plot
        title : str
            title for the scatter plot
    """
    plt.figure(figsize=(20, 10))
    cluster_labels = clustering_method.fit_predict(df)
    colors = get_array_random_colors(len(np.unique(cluster_labels)) - 1)
    cluster_colors = []

    i = 0
    while i < len(cluster_labels):
        if(cluster_labels[i] == -1):
            # noise should be black
            cluster_colors.append("#000000")
        else:
            cluster_colors.append(colors[cluster_labels[i]])
        i += 1

    plt.scatter(df[0], df[1], c=cluster_colors, cmap='Paired')
    plt.title(title)
    plt.show()


def create_3d_scatterplot_clustering(clustering_method, df, title):
    """ create 3D clustering scatter plot from given dataFrame

        Parameters
        ----------
        clustering_method : object
            clustering method (before being fit to the dataset)
        df : dataFrame
            data for the scatter plot
        title : str
            title for the scatter plot
    """
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
    """ create 2D pyplot (for 4-dist graph) from given data

        Parameters
        ----------
        data : list
            data for the pyplot
    """
    plt.figure(figsize=(20, 10))
    plt.plot(data, "ro")
    plt.title('n-dist sorted graph')
    plt.ylabel('distance to 4th nearest neighbor')
    plt.show()


def get_array_random_colors(size):
     """ create array of size number of random hex colors

        Parameters
        ----------
        size : int
            number of random colors to return
        Returns
        -------
        list (array) of random hex colors
    """
    random.seed(70)
    colors = set()
    i = 0
    while i < size:
        random_color = "#" + "%06x" % random.randint(0, 0xFFFFFF)
        # to avoid duplicate colors
        while(random_color in colors):
            random_color = "#" + "%06x" % random.randint(0, 0xFFFFFF)
        colors.add(random_color)
        i += 1
    return list(colors)


def create_reachability_plot(df, clustering, eps_line=False):
    """ create reachability bar plot from OPTICS results

        Parameters
        ----------
        df : dataFrame
            data for the plot
        clustering : object
            OPTICS clustering results for df
        eps_line : Boolean
            render eps_line in plot (default 2.0 - to match DBSCAN)
    """
    space = np.arange(len(df))
    reachability = clustering.reachability_[clustering.ordering_]
    labels = clustering.labels_[clustering.ordering_]
    plt.figure(figsize=(50, 10))

    unique_labels = np.unique(labels)

    ax1 = plt.subplot()

    # Plotting the Reachability-Distance Plot
    colors = get_array_random_colors(len(unique_labels) - 1)

    # add color to the points in a cluster
    for Class, color in zip(range(0, unique_labels[len(unique_labels) - 1]), colors):
        Xk = space[labels == Class]
        Rk = reachability[labels == Class]
        ax1.bar(Xk, Rk, color=color, width=1)
    ax1.bar(space[labels == -1], reachability[labels == -1],
            color='k', width=1)

    # add the points in no cluster (noise) with the color black
    if eps_line:
        ax1.plot(space, np.full_like(
            space, 2, dtype=float), color='k', alpha=0.5)

    ax1.fill_between(0, space)
    ax1.set_ylim([0, 7.5])
    ax1.set_ylabel('Reachability Distance')
    ax1.set_title('Reachability Plot')
    plt.show()
