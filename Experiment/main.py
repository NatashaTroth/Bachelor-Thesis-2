import pandas as pd
# import numpy as np
from dataFile import DataFile
from dataFiles import DataFiles
from clusterEvaluation import compare_scores
from pathlib import Path
import sys

number_dimensions = 2


def print_mean_of_two_scores(scores1, scores2):
    i = 0
    while i < 3:
        print((scores1[i] + scores2[i])/2)
        i += 1


# ---- ONE HOUR FILES ----
print("----1 HOUR FILE----")
one_hour_file = DataFile(
    "/Volumes/BATroth/aggregated/1h")

# ---clean data (preprocessing)---
one_hour_file.clean_data(1)

# ---dimensionality reductions---
one_hour_file.calculate_TSNE(number_dimensions, False)

# ---clustering---
one_hour_file.dbscan_clustering('TSNE', False)
one_hour_file.optics_clustering('TSNE', True)

# # ---- ONE HOUR FILES ----
# print("\n----1 HOUR FILE 2----")
# one_hour_file_2 = DataFile(
#     "/Volumes/BATroth/aggregated/1h")

# # ---clean data (preprocessing)---
# one_hour_file_2.clean_data(5)

# # ---dimensionality reductions---
# one_hour_file_2.calculate_TSNE(number_dimensions, False)

# # ---clustering---
# one_hour_file_2.dbscan_clustering('TSNE', False)
# one_hour_file_2.optics_clustering('TSNE', False)
# # ---------------------------------------------
# # ---------------------------------------------

print("\n------------------------")
print("One hour results 1:")
print("------------------------")
print("dbscan:")
print(one_hour_file.dbscan_scores)
print("optics:")
print(one_hour_file.optics_scores)

# print("\n------------------------")
# print("One hour results 2:")
# print("------------------------")
# print("dbscan:")
# print(one_hour_file_2.dbscan_scores)
# print("optics:")
# print(one_hour_file_2.optics_scores)

# print("-------mean 1 hour file:------")
# print("DBSCAN: ")
# print_mean_of_two_scores(one_hour_file.dbscan_scores,
#                          one_hour_file_2.dbscan_scores)
# print("OPTICS: ")
# print_mean_of_two_scores(one_hour_file.optics_scores,
#                          one_hour_file_2.optics_scores)


# ---- THREE HOUR FILES ----
print("\n----3 HOUR FILE----")
three_hour_file = DataFile(
    "/Volumes/BATroth/aggregated/3h")

# ---clean data (preprocessing)---
three_hour_file.clean_data(1)

# ---dimensionality reductions---
three_hour_file.calculate_TSNE(number_dimensions, False)

# ---clustering---
three_hour_file.dbscan_clustering('TSNE', False)
three_hour_file.optics_clustering('TSNE', True)


# # ---- THREE HOUR FILES ----
# print("\n----3 HOUR FILE 2----")
# three_hour_file_2 = DataFile(
#     "/Volumes/BATroth/aggregated/3h")

# # ---clean data (preprocessing)---
# three_hour_file_2.clean_data(6)

# # ---dimensionality reductions---
# three_hour_file_2.calculate_TSNE(number_dimensions, False)

# # ---clustering---
# three_hour_file_2.dbscan_clustering('TSNE', False)
# three_hour_file_2.optics_clustering('TSNE', False)

# ---------------------------------------------
# ---------------------------------------------
print("\n------------------------")
print("Three hour results 1:")
print("------------------------")
print("dbscan:")
print(three_hour_file.dbscan_scores)
print("optics:")
print(three_hour_file.optics_scores)


# print("\n------------------------")
# print("Three hour results 2:")
# print("------------------------")
# print("dbscan:")
# print(three_hour_file_2.dbscan_scores)
# print("optics:")
# print(three_hour_file_2.optics_scores)

# print("-------mean 3 hour file:------")
# print("DBSCAN: ")
# print_mean_of_two_scores(three_hour_file.dbscan_scores,
#                          three_hour_file_2.dbscan_scores)
# print("OPTICS: ")
# print_mean_of_two_scores(three_hour_file.optics_scores,
#                          three_hour_file_2.optics_scores)


# # ----- Compare Results -----
# print("\n\n----Compare Results----")
# result_dbscan = compare_scores(
#     one_hour_file.dbscan_scores, three_hour_file.dbscan_scores)
# print("result_dbscan " + str(result_dbscan))
# result_optics = compare_scores(
#     one_hour_file.optics_scores, three_hour_file.optics_scores)
# print("result_optics " + str(result_optics))
# # result_agglomerative = compare_scores(
# #     one_hour_file.agglomerative_scores, three_hour_file.agglomerative_scores)
# # print("result_agglomerative " + str(result_agglomerative))


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # oneHourFile = DataFile(
# # "/Users/natashatroth/Documents/FHS/6Semester/Bac2/testData/1h")

# # ---- ONE HOUR FILES ----
# print("----1 HOUR FILE----")
# one_hour_file = DataFile(
#     "/Volumes/BATroth/aggregated/1h")


# # ---clean data (preprocessing)---
# # one_hour_file.print_file()
# one_hour_file.clean_data(1)

# # one_hour_file.print_file()

# # ---dimensionality reductions---
# # one_hour_file.calculate_PCA(number_dimensions, True)
# # one_hour_file.calculate_ideal_TSNE_perplexity()
# one_hour_file.calculate_TSNE(number_dimensions, False)

# # ---clustering---
# # one_hour_file.spectral_clustering('PCA')
# # one_hour_file.spectral_clustering('TSNE')

# # one_hour_file.dbscan_clustering('PCA', True)
# one_hour_file.dbscan_clustering('TSNE', False)
# # one_hour_file.dbscan_clustering('', True)


# # one_hour_file.optics_clustering('PCA', True)
# one_hour_file.optics_clustering('TSNE', False)
# # one_hour_file.optics_clustering('', True)

# # one_hour_file.agglomerative_clustering('PCA', True)
# # one_hour_file.agglomerative_clustering('TSNE', True)
# # one_hour_file.agglomerative_clustering('', True)


# # ---- THREE HOUR FILES ----
# print("\n----3 HOUR FILE----")
# three_hour_file = DataFile(
#     "/Volumes/BATroth/aggregated/3h")

# # # ---clean data (preprocessing)---
# # # three_hour_file.print_file()
# three_hour_file.clean_data(1)
# # # three_hour_file.print_file()

# # # ---dimensionality reductions---
# # three_hour_file.calculate_PCA(number_dimensions, False)
# three_hour_file.calculate_TSNE(number_dimensions, False)

# # # ---clustering---
# # # three_hour_file.spectral_clustering('PCA')
# # three_hour_file.spectral_clustering('TSNE')

# # three_hour_file.dbscan_clustering('PCA', True)
# three_hour_file.dbscan_clustering('TSNE', False)
# # three_hour_file.dbscan_clustering('', True)

# # three_hour_file.optics_clustering('PCA', True)
# three_hour_file.optics_clustering('TSNE', False)
# # three_hour_file.optics_clustering('', True)

# # three_hour_file.agglomerative_clustering('PCA', True)
# # three_hour_file.agglomerative_clustering('TSNE', False)
# # three_hour_file.agglomerative_clustering('', True)


# # ----- Compare Results -----
# print("\n\n----Compare Results----")
# result_dbscan = compare_scores(
#     one_hour_file.dbscan_scores, three_hour_file.dbscan_scores)
# print("result_dbscan " + str(result_dbscan))
# result_optics = compare_scores(
#     one_hour_file.optics_scores, three_hour_file.optics_scores)
# print("result_optics " + str(result_optics))
# # result_agglomerative = compare_scores(
# #     one_hour_file.agglomerative_scores, three_hour_file.agglomerative_scores)
# # print("result_agglomerative " + str(result_agglomerative))
