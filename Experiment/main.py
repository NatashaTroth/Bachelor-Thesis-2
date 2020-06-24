from dataFile import DataFile
from clusterEvaluation import compare_scores


def create_clustering_of_directory(directory, number_columns_to_use=1, number_dimensions=2):
    data_file = DataFile(directory)

    # ---clean data (preprocessing)---
    data_file.clean_data(number_columns_to_use)

    # ---dimensionality reductions---
    data_file.calculate_PCA(number_dimensions, True)
    # data_file.calculate_TSNE(number_dimensions, True)

    # ---clustering---
    # data_file.dbscan_clustering('PCA', True)
    # data_file.dbscan_clustering('TSNE', True)
    # data_file.dbscan_clustering('', True)

    # data_file.optics_clustering('PCA', True)
    # data_file.optics_clustering('TSNE', True)
    # data_file.optics_clustering('', True)

    # ---other clustering methods---
    # data_file.spectral_clustering('PCA')
    # data_file.spectral_clustering('TSNE')

    # data_file.agglomerative_clustering('PCA', True)
    # data_file.agglomerative_clustering('TSNE', True)
    # data_file.agglomerative_clustering('', True)

    return data_file


def print_mean_of_two_scores(scores1, scores2):
    i = 0
    while i < 3:
        print((scores1[i] + scores2[i])/2)
        i += 1


def print_average_scores(data_file_1, data_file_2):
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
