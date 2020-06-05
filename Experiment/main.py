import pandas as pd
# import numpy as np
from dataFile import DataFile
from dataFiles import DataFiles
from pathlib import Path
import sys


# oneHourFile = DataFile(
#     "/Users/natashatroth/Documents/FHS/6Semester/Bac2/testData/1h")

# ---- ONE HOUR FILES ----
one_hour_file = DataFile(
    "/Volumes/BATroth/aggregated/1h")

# ---clean data (preprocessing)---
# one_hour_file.print_file()
one_hour_file.clean_data(4)
# one_hour_file.print_file()

# ---dimensionality reductions---
one_hour_file.calculate_PCA(2, True)
one_hour_file.calculate_TSNE(2, True)

# ---clustering---
# one_hour_file.spectral_clustering('PCA')
# one_hour_file.spectral_clustering('TSNE')

# one_hour_file.dbscan_clustering('PCA')
one_hour_file.dbscan_clustering('TSNE')


# # ---- THREE HOUR FILES ----
# three_hour_file = DataFile(
#     "/Volumes/BATroth/aggregated/1h")

# # ---clean data (preprocessing)---
# # three_hour_file.print_file()
# three_hour_file.clean_data(4)
# # three_hour_file.print_file()

# # ---dimensionality reductions---
# three_hour_file.calculate_PCA(False)
# three_hour_file.calculate_TSNE(2, False)

# # ---clustering---
# # three_hour_file.spectral_clustering('PCA')
# # three_hour_file.spectral_clustering('TSNE')

# # three_hour_file.dbscan_clustering('PCA')
# three_hour_file.dbscan_clustering('TSNE')
