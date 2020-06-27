from dataFile import DataFile
from clusterEvaluation import compare_scores


def create_clustering_of_directory(directory, number_columns_to_use=1, number_dimensions=2):
    """executes data cleaning, dimensionality reduction, and clustering of a file
       comment in/out what types of dimensionality reduction and clustering should be used

    Parameters
    ----------
    directory : str
       The directory where the .csv files are located
    number_columns_to_use : int, optional
       Number of columns (1-N) to average and calculate with (default 1)
    number_dimensions : int, optional
       Number of dimensions to reduce the data to (default 2)

    Returns
    -------
    dataFile object
    """

    data_file = DataFile(directory)

    # ---clean data (preprocessing)---
    data_file.clean_data(number_columns_to_use)

    # ---dimensionality reductions---
    data_file.calculate_TSNE(number_dimensions, True)
    # data_file.calculate_PCA(number_dimensions, True)

    # ---clustering---
    data_file.dbscan_clustering('TSNE', True)
    # data_file.dbscan_clustering('PCA', True)
    # data_file.dbscan_clustering('', True)

    data_file.optics_clustering('TSNE', True)
    # data_file.optics_clustering('PCA', True)
    # data_file.optics_clustering('', True)

    # ---other clustering methods---
    # data_file.spectral_clustering('TSNE')
    # data_file.spectral_clustering('PCA')

    # data_file.agglomerative_clustering('PCA', True)
    # data_file.agglomerative_clustering('TSNE', True)
    # data_file.agglomerative_clustering('', True)

    return data_file


def print_mean_of_two_scores(scores1, scores2):
    """print the mean of two scores (floats)

    Parameters
    ----------
    score 1 : int
    score 2 : int
    """
    i = 0
    while i < 3:
        print((scores1[i] + scores2[i])/2)
        i += 1


def print_average_scores(data_file_1, data_file_2):
    """print the average evaluations scores of two data files

    Parameters
    ----------
    data_file_1 : dataFile
    data_file_2 : dataFile
    """
    print("\n------------------------")
    print("Results data file 1:")
    print("------------------------")
    print("dbscan:")
    print(data_file_1.dbscan_scores)
    print("optics:")
    print(data_file_1.optics_scores)

    print("\n------------------------")
    print("Results data file 2:")
    print("------------------------")
    print("dbscan:")
    print(data_file_2.dbscan_scores)
    print("optics:")
    print(data_file_2.optics_scores)

    print("-------mean 1 hour file:------")
    print("DBSCAN: ")
    print_mean_of_two_scores(data_file_1.dbscan_scores,
                             data_file_2.dbscan_scores)
    print("OPTICS: ")
    print_mean_of_two_scores(data_file_1.optics_scores,
                             data_file_2.optics_scores)


def main():
    print("----1 HOUR FILES----")
    one_hour_file = create_clustering_of_directory(
        "/Volumes/BATroth/aggregated/1h", 1)

    print("----3 HOUR FILES----")
    three_hour_file = create_clustering_of_directory(
        "/Volumes/BATroth/aggregated/3h", 1)

    # ----- Average of two files -----
    # print("----1 HOUR FILES 2----")
    # one_hour_file_2 = create_clustering_of_directory(
    #     "/Volumes/BATroth/aggregated/1h", 1)
    # print_average_scores(one_hour_file, one_hour_file_2)

    # print("----3 HOUR FILES 2----")
    # three_hour_file_2 = create_clustering_of_directory(
    #     "/Volumes/BATroth/aggregated/3h", 1)
    # print_average_scores(three_hour_file, three_hour_file_2)

    # ----- Compare Results -----
    print("\n\n----Compare Results----")
    result_dbscan = compare_scores(
        one_hour_file.dbscan_scores, three_hour_file.dbscan_scores)
    print("result_dbscan " + str(result_dbscan))
    result_optics = compare_scores(
        one_hour_file.optics_scores, three_hour_file.optics_scores)
    print("result_optics " + str(result_optics))


if __name__ == "__main__":
    main()
