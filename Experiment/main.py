import pandas as pd
# import numpy as np
from dataFile import DataFile
from dataFiles import DataFiles
from pathlib import Path
import sys
from clustering import calculate_PCA
from clustering import calculate_TSNE


# oneHourFiles = DataFiles(
#     "/Volumes/BATroth/aggregated/1h")
# "/Users/natashatroth/Documents/FHS/6Semester/Bac2/testData/1h")
# print(oneHourFiles.dataFiles)
# oneHourFiles.replaceNonIntWithNaN()


# oneHourFile = DataFile(
#     "/Users/natashatroth/Documents/FHS/6Semester/Bac2/testData/1h")

# ONE HOUR FILES
one_hour_file = DataFile(
    "/Volumes/BATroth/aggregated/1h")

# one_hour_file.print_file()
one_hour_file.clean_data(4)
# one_hour_file.print_file()
# calculate_PCA(one_hour_file.df)
calculate_TSNE(one_hour_file.df)


# THREE HOUR FILES
three_hour_file = DataFile(
    "/Volumes/BATroth/aggregated/3h")

three_hour_file.print_file()
three_hour_file.clean_data(6)
three_hour_file.print_file()
# calculate_PCA(three_hour_file.df)
calculate_TSNE(three_hour_file.df)
