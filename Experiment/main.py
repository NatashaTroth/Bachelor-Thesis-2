import pandas as pd
# import numpy as np
from dataFile import DataFile
from dataFiles import DataFiles
from pathlib import Path
import sys

# oneHourFiles = DataFiles(
#     "/Volumes/BATroth/aggregated/1h")
# "/Users/natashatroth/Documents/FHS/6Semester/Bac2/testData/1h")
# print(oneHourFiles.dataFiles)
# oneHourFiles.replaceNonIntWithNaN()


# oneHourFile = DataFile(
#     "/Users/natashatroth/Documents/FHS/6Semester/Bac2/testData/1h")
oneHourFile = DataFile(
    "/Volumes/BATroth/aggregated/1h")

oneHourFile.printFile()
oneHourFile.cleanData()
oneHourFile.saveFile()
oneHourFile.printFile()


# oneHourFiles.printFiles()
# oneHourFiles.cleanData()
# oneHourFiles.printFiles()


# print(oneHourFiles.dataFiles[0].df["NOTIF1"])


# threeHourFiles = DataFiles(
#     "/Users/natashatroth/Documents/FHS/6Semester/Bac2/testData/3h")
# print(threeHourFiles.dataFiles)

# pathlist = Path(
#     "/Users/natashatroth/Documents/FHS/6Semester/Bac2/testData/1h").glob('**/*.csv')
# for path in pathlist:
#     path_in_str = str(path)
#     print(path_in_str)


# firstFile = DataFile(
#     "/Users/natashatroth/Documents/FHS/6Semester/Bac2/testData/testData.csv")


# print(firstFile.df["NOTIF1"])
# firstFile.replaceAllNotIntWithNaN()
# print(firstFile.df["NOTIF1"])


# print(firstFile.df.head())
# print(firstFile.df["NOTIF1"])
# print(firstFile.df["NOTIF1"].fillna(6))
# print(firstFile.df["NOTIF1"].interpolate(
#     method='linear').interpolate(method='linear', limit_direction='backward', limit=1))
# print(df["NOTIF1"])
# print(df["NOTIF1"].isnull())


# df = pd.read_csv(
#     "/Users/natashatroth/Documents/FHS/6Semester/Bac2/testData/testData.csv")
