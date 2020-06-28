from sklearn.cluster import SpectralClustering, DBSCAN, OPTICS, AgglomerativeClustering
from plot import create_2d_scatterplot_clustering, create_clustering_plot, create_2d_pyplot, create_reachability_plot
from sklearn.neighbors import NearestNeighbors
from clusterEvaluation import cluster_evaluation


def dbscan_clustering(df, dataType="", graphs=False):
    """ apply DBSCAN clustering algorithm to the dataFrame

        Parameters
        ----------
        df : dataFrame
            dataFrame object with the data to be reduced
        data_type : str
            type of data from the dataFrame to feed into the clustering algorithm (options: "PCA", "t-SNE", or "" which means the entire dataFrame)
        graphs : Boolean
            render clustering scatter plot graphs (True), or not (False)

        Returns
        -------
        list of the evaluation score results ([silhouette_score_evaluation, davies_bouldin_score_evaluation, calinski_harabasz_score_evaluation])
    """
    print("\n- dbscan clustering (" + dataType + ")...")
    db = DBSCAN(eps=2, min_samples=4)
    clustering = db.fit(df)

    if graphs == True:
        create_clustering_plot(db, df, "DBSCAN (" + dataType + ")")
    return cluster_evaluation(db, df)


def predict_eps_dbscan_parameter(df):
    """ predict DBSCAN Eps parameter using 4-dist graph

        Parameters
        ----------
        df : dataFrame
            dataFrame object with the data to be reduced
    """
    print("predict Eps DBSCAN parameter...")
    nbrs = NearestNeighbors(
        n_neighbors=4, algorithm='ball_tree', metric='euclidean')
    nbrsResult = nbrs.fit(df)
    distance, ind = nbrs.kneighbors(df)

    # dist is a 2 dimensional array with the rows and distances (each row is list of length 4 - distances to 4 nearest neighbours).
    # get distances to only the 4th nearest neighbour
    distances = [distance[i][3] for i in range(len(distance))]
    distances.sort(reverse=True)
    create_2d_pyplot(distances)


def optics_clustering(df, dataType="", graphs=False):
    """ apply OPTICS clustering algorithm to the dataFrame

        Parameters
        ----------
        df : dataFrame
            dataFrame object with the data to be reduced
        data_type : str
            type of data from the dataFrame to feed into the clustering algorithm (options: "PCA", "t-SNE", or "" which means the entire dataFrame)
        graphs : Boolean
            render clustering scatter plot graphs (True), or not (False)

        Returns
        -------
        list of the evaluation score results ([silhouette_score_evaluation, davies_bouldin_score_evaluation, calinski_harabasz_score_evaluation])
    """
    print("\n- optics clustering (" + dataType + ")...")
    optics = OPTICS(cluster_method='dbscan', eps=2)
    # optics = OPTICS(min_samples=4, cluster_method="xi")
    clustering = optics.fit(df)
    if graphs == True:
        create_clustering_plot(
            optics, df, "OPTICS (" + dataType + ")")
        create_reachability_plot(df, clustering, True)
    return cluster_evaluation(optics, df)


# ---------- OTHER CLUSTERING METHODS ----------

def spectral_clustering(df, dataType="", graphs=False):
    """ apply spectral clustering algorithm to the dataFrame

        Parameters
        ----------
        df : dataFrame
            dataFrame object with the data to be reduced
        data_type : str
            type of data from the dataFrame to feed into the clustering algorithm (options: "PCA", "t-SNE", or "" which means the entire dataFrame)
        graphs : Boolean
            render clustering scatter plot graphs (True), or not (False)

        Returns
        -------
        list of the evaluation score results ([silhouette_score_evaluation, davies_bouldin_score_evaluation, calinski_harabasz_score_evaluation])
    """
    print("\n- spectral clustering (" + dataType + ")...")
    sc = SpectralClustering(
        n_clusters=3, assign_labels="discretize", random_state=0)
    clustering = sc.fit(df)
    if graphs == True:
        create_clustering_plot(
            sc, df, "Spectral Clustering (" + dataType + ")")
    return cluster_evaluation(sc, df)


def agglomerative_clustering(df, dataType="", graphs=False):
    """ apply agglomerative clustering algorithm to the dataFrame

        Parameters
        ----------
        df : dataFrame
            dataFrame object with the data to be reduced
        data_type : str
            type of data from the dataFrame to feed into the clustering algorithm (options: "PCA", "t-SNE", or "" which means the entire dataFrame)
        graphs : Boolean
            render clustering scatter plot graphs (True), or not (False)

        Returns
        -------
        list of the evaluation score results ([silhouette_score_evaluation, davies_bouldin_score_evaluation, calinski_harabasz_score_evaluation])
    """
    print("\n- agglomerative clustering (" + dataType + ")...")
    agglomerative = AgglomerativeClustering()
    clustering = agglomerative.fit(df)
    if graphs == True:
        create_clustering_plot(
            agglomerative, df, "Agglomerative Clustering (" + dataType + ")")
    return cluster_evaluation(agglomerative, df)
