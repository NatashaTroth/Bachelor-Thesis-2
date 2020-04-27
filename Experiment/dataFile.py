import pandas as pd
import numpy as np

# TODO: index data https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html


class DataFile:
    def __init__(self, filePath):
        self.df = pd.read_csv(filePath)
        self.filePath = filePath

    def printFile(self):
        print(self.df)

    def replaceNonIntWithNaN(self):
        # print("FILE:" + self.filePath)
        for column in self.df:
            if(not column.startswith("TIME")):
                self.replaceNonIntWithNaNPerColumn(column)

    def replaceNonIntWithNaNPerColumn(self, column):
        counter = 0
        # print("COLOUMN:" + column)
        for row in self.df[column]:
            if(not column.startswith("TIME")):

                # print(row)
                try:
                    # test if value can be converted to an int - if not, it is not a number
                    float(row)
                    self.df.loc[counter, column] = float(row)
                    pass
                except ValueError:
                    self.df.loc[counter, column] = np.nan
                counter += 1

# ------------REPLACE MISSING VALUES-------------
    def replaceMissingValuesZero(self):
        print("in replace missing values zero")
        for column in self.df:
            if(not column.startswith("TIME")):
                self.df[column].fillna(0, inplace=True)

    def replaceMissingValuesInterpolate(self, method):
        print("in replace missing values interpolate")
        for column in self.df:
            if(not column.startswith("TIME")):

                self.df[column].interpolate(method=method, inplace=True)
                self.df[column].interpolate(
                    method=method, inplace=True, limit_direction='backward', limit=1)

    def replaceMissingValueMean(self):
        print("in replace missing values mean")
        for column in self.df:
            if(not column.startswith("TIME")):
                mean = self.df[column].mean()
                self.df[column].fillna(mean, inplace=True)

    def replaceMissingValueMedian(self):
        print("in replace missing values median")
        for column in self.df:
            if(not column.startswith("TIME")):
                median = self.df[column].median()
                self.df[column].fillna(median, inplace=True)

    def replaceMissingValuesCustom(self):
        print("in replace missing values custom")
        for column in self.df:
          # skipped Time
            if(column.startswith("ACC")):
                self.df[column].fillna(6, inplace=True)
            if(column.startswith("AUDIO")):
                self.df[column].fillna(6, inplace=True)
            if(column.startswith("SCRN")):
                self.df[column].fillna(6, inplace=True)
            if(column.startswith("NOTIF")):
                self.df[column].fillna(6, inplace=True)
            if(column.startswith("LIGHT")):
                self.df[column].fillna(6, inplace=True)
            if(column.startswith("APP_VID")):
                self.df[column].fillna(6, inplace=True)
            if(column.startswith("APP_COMM")):
                self.df[column].fillna(6, inplace=True)
            if(column.startswith("APP_OTHER")):
                self.df[column].fillna(6, inplace=True)

    # def replaceMissingValuesPerColumn(self, column):
    #     counter = 0
    #     for row in self.df[column]:
    #         try:

    #             int(row)
    #             pass
    #         except ValueError:
    #             self.df.loc[counter, column] = np.nan
    #         counter += 1
        # print df.isnull().values.any()
