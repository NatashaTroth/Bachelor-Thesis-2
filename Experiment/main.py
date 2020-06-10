import pandas as pd
# import numpy as np
from dataFile import DataFile
from dataFiles import DataFiles
from pathlib import Path
import sys

number_dimensions = 2

# oneHourFile = DataFile(
#     "/Users/natashatroth/Documents/FHS/6Semester/Bac2/testData/1h")

# ---- ONE HOUR FILES ----
print("----1 HOUR FILE----")
one_hour_file = DataFile(
    "/Volumes/BATroth/aggregated/1h")

# ---clean data (preprocessing)---
# one_hour_file.print_file()
one_hour_file.clean_data(4)

# one_hour_file.print_file()

# ---dimensionality reductions---
# one_hour_file.calculate_PCA(number_dimensions, False)
one_hour_file.calculate_TSNE(number_dimensions, True)

# ---clustering---
# one_hour_file.spectral_clustering('PCA')
# one_hour_file.spectral_clustering('TSNE')

# one_hour_file.dbscan_clustering('PCA', True)
one_hour_file.dbscan_clustering('TSNE', True)
# one_hour_file.dbscan_clustering('', True)

# one_hour_file.optics_clustering('PCA', True)
one_hour_file.optics_clustering('TSNE', True)
# one_hour_file.optics_clustering('', True)

# one_hour_file.agglomerative_clustering('PCA', True)
# one_hour_file.agglomerative_clustering('TSNE', True)
# one_hour_file.agglomerative_clustering('', True)


# # ---- THREE HOUR FILES ----
# print("\n----3 HOUR FILE----")
# three_hour_file = DataFile(
#     "/Volumes/BATroth/aggregated/3h")

# # # ---clean data (preprocessing)---
# # # three_hour_file.print_file()
# three_hour_file.clean_data(6)
# # # three_hour_file.print_file()

# # # ---dimensionality reductions---
# # three_hour_file.calculate_PCA(number_dimensions, False)
# three_hour_file.calculate_TSNE(number_dimensions, True)

# # # ---clustering---
# # # three_hour_file.spectral_clustering('PCA')
# # three_hour_file.spectral_clustering('TSNE')

# # three_hour_file.dbscan_clustering('PCA', True)
# three_hour_file.dbscan_clustering('TSNE', True)
# # three_hour_file.dbscan_clustering('', True)

# # three_hour_file.optics_clustering('PCA', True)
# three_hour_file.optics_clustering('TSNE', True)
# # three_hour_file.optics_clustering('', True)

# # three_hour_file.agglomerative_clustering('PCA', True)
# three_hour_file.agglomerative_clustering('TSNE', True)
# # three_hour_file.agglomerative_clustering('', True)
