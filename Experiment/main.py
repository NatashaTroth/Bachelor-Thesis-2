import pandas as pd
# import numpy as np
from dataFile import DataFile
from dataFiles import DataFiles
from pathlib import Path
import sys
from clustering import calculatePCA


# oneHourFiles = DataFiles(
#     "/Volumes/BATroth/aggregated/1h")
# "/Users/natashatroth/Documents/FHS/6Semester/Bac2/testData/1h")
# print(oneHourFiles.dataFiles)
# oneHourFiles.replaceNonIntWithNaN()


# oneHourFile = DataFile(
#     "/Users/natashatroth/Documents/FHS/6Semester/Bac2/testData/1h")

# ONE HOUR FILES
oneHourFile = DataFile(
    "/Volumes/BATroth/aggregated/1h")

oneHourFile.printFile()
oneHourFile.cleanData(4)
# calculatePCA(oneHourFile.df)
oneHourFile.compressSameAttributeColumns(4)
oneHourFile.printFile()
calculatePCA(oneHourFile.df)


# THREE HOUR FILES
threeHourFile = DataFile(
    "/Volumes/BATroth/aggregated/3h")

threeHourFile.printFile()
threeHourFile.cleanData(6)
threeHourFile.compressSameAttributeColumns(6)
threeHourFile.printFile()
calculatePCA(threeHourFile.df)
