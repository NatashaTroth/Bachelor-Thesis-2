import pandas as pd
# import numpy as np
from dataFile import DataFile

firstFile = DataFile(
    "/Users/natashatroth/Documents/FHS/6Semester/Bac2/testData/testData.csv")

# df = pd.read_csv(
#     "/Users/natashatroth/Documents/FHS/6Semester/Bac2/testData/testData.csv")

print(firstFile.df["NOTIF1"])
firstFile.replaceAllNotIntWithNaN()
print(firstFile.df["NOTIF1"])


# print(firstFile.df.head())
# print(firstFile.df["NOTIF1"])
# print(firstFile.df["NOTIF1"].fillna(6))
# print(firstFile.df["NOTIF1"].interpolate(
#     method='linear').interpolate(method='linear', limit_direction='backward', limit=1))
# print(df["NOTIF1"])
# print(df["NOTIF1"].isnull())
