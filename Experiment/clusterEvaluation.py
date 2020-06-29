import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def cluster_evaluation(clustering_method, df):
    """ return Silhouette Score, Davies Bouldin Score, Calinski Harabasz Score in a list

        Parameters
        ----------
        clustering_method : object
            clustering method (before being fit to the dataset)
        df : dataFrame
            dataFrame object with the data to be reduced

        Returns
        -------
        list of the evaluation score results ([silhouette_score_evaluation, davies_bouldin_score_evaluation, calinski_harabasz_score_evaluation])
    """
    cluster_evaluation_scores = []
    cluster_evaluation_scores.append(silhouette_score_evaluation(
        clustering_method, df))
    cluster_evaluation_scores.append(davies_bouldin_score_evaluation(
        clustering_method, df))
    cluster_evaluation_scores.append(calinski_harabasz_score_evaluation(
        clustering_method, df))
    return cluster_evaluation_scores


def silhouette_score_evaluation(clustering_method, df):
    """ calculate overall average silhouette width for the dataFrame

        Parameters
        ----------
        clustering_method : object
            clustering method (before being fit to the dataset)
        df : dataFrame
            dataFrame object with the data to be reduced

        Returns
        -------
        silhouette score as float
    """
    cluster_labels = clustering_method.fit_predict(df)
    if len(np.unique(cluster_labels)) > 1:
        score = silhouette_score(df, cluster_labels)
        print("Silhouette score: " + str(score))
        return score
    else:
        print("Only one cluster - there have to be at least 2 clusters to calculation the Silhouette score.")
        return None


def davies_bouldin_score_evaluation(clustering_method, df):
    """ calculate overall davies-bouldin index for the dataFrame

        Parameters
        ----------
        clustering_method : object
            clustering method (before being fit to the dataset)
        df : dataFrame
            dataFrame object with the data to be reduced

        Returns
        -------
        davies-bouldin index as float
    """
    cluster_labels = clustering_method.fit_predict(df)

    if len(np.unique(cluster_labels)) > 1:
        score = davies_bouldin_score(df, cluster_labels)
        print("Davies Bouldin score: " + str(score))
        return score
    else:
        print("Only one cluster - there have to be at least 2 clusters to calculation the Davies Bouldin score.")
        return None


def calinski_harabasz_score_evaluation(clustering_method, df):
    """ calculate overall calinski-harabasz index for the dataFrame

        Parameters
        ----------
        clustering_method : object
            clustering method (before being fit to the dataset)
        df : dataFrame
            dataFrame object with the data to be reduced

        Returns
        -------
        calinski-harabasz index as float
    """
    cluster_labels = clustering_method.fit_predict(df)
    if len(np.unique(cluster_labels)) > 1:
        score = calinski_harabasz_score(df, cluster_labels)
        print("Calinski Harabasz score: " + str(score))
        return score
    else:
        print("Only one cluster - there have to be at least 2 clusters to calculation the Calinski Harabasz score.")
        return None
