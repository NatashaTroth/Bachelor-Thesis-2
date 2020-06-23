import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def cluster_evaluation(clustering_method, df):
    cluster_evaluation_scores = []
    cluster_evaluation_scores.append(silhouette_score_evaluation(
        clustering_method, df))
    cluster_evaluation_scores.append(davies_bouldin_score_evaluation(
        clustering_method, df))
    cluster_evaluation_scores.append(calinski_harabasz_score_evaluation(
        clustering_method, df))
    return cluster_evaluation_scores


def silhouette_score_evaluation(clustering_method, df):
    cluster_labels = clustering_method.fit_predict(df)
    if len(np.unique(cluster_labels)) > 1:
        score = silhouette_score(df, cluster_labels)
        print("Silhouette score: " + str(score))
        return score
    else:
        print("Only one cluster - there have to be at least 2 clusters to calculation the Silhouette score.")
        return None


def davies_bouldin_score_evaluation(clustering_method, df):
    cluster_labels = clustering_method.fit_predict(df)

    if len(np.unique(cluster_labels)) > 1:
        score = davies_bouldin_score(df, cluster_labels)
        print("Davies Bouldin score: " + str(score))
        return score
    else:
        print("Only one cluster - there have to be at least 2 clusters to calculation the Davies Bouldin score.")
        return None


def calinski_harabasz_score_evaluation(clustering_method, df):
    cluster_labels = clustering_method.fit_predict(df)
    if len(np.unique(cluster_labels)) > 1:
        score = calinski_harabasz_score(df, cluster_labels)
        print("Calinski Harabasz score: " + str(score))
        return score
    else:
        print("Only one cluster - there have to be at least 2 clusters to calculation the Calinski Harabasz score.")
        return None


# returns 1 if first scores are better, returns 2 if second scores are better, returns 0 if tie
def compare_scores(scores_1, scores_2):
    scores_1_wins = 0
    scores_2_wins = 0
    # silhouette score (score closer to 1 better, closer to -1 not so good - higher values better)
    if scores_1[0] is not None and scores_2[0] is not None:
        if scores_1[0] > scores_2[0]:
            scores_1_wins += 1
        elif scores_1[0] < scores_2[0]:
            scores_2_wins += 1

    # davies bouldin score (zero is the lowest possible score. Values closer to zero indicate a better partition - smaller values better)
    if scores_1[1] is not None and scores_2[1] is not None:
        if scores_1[1] < scores_2[1]:
            scores_1_wins += 1
        elif scores_1[1] > scores_2[1]:
            scores_2_wins += 1

    # calinksi harabasz score (higher score relates to a model with better defined clusters - higher values better)
    if scores_1[2] is not None and scores_2[2] is not None:
        if scores_1[2] > scores_2[2]:
            scores_1_wins += 1
        elif scores_1[2] < scores_2[2]:
            scores_2_wins += 1

    # results
    if scores_1_wins > scores_2_wins:
        return 1
    elif scores_1_wins < scores_2_wins:
        return 2
    else:
        return 0
