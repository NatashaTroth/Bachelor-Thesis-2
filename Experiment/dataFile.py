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

    def removeEmptyRows(self):
        print("in remove empty rows")
        self.df.dropna(axis=0, how='all',  inplace=True)

        # thresh=None,subset=None,
        # Axis: Specifies to drop by row or column. 0 means row, 1 means column.
        # How: Accepts one of two possible values: any or all. This will either drop an axis which is completely empty (all), or an axis with even just a single empty cell (any).
        # Thresh: Here's an interesting one: thresh accepts an integer, and will drop an axis only if that number threshold of empty cells is breached.
        # Subset: Accepts an array of which axis' to consider, as opposed to considering all by default.
        # Inplace: If you haven't come across inplace yet, learn this now: changes will NOT be made to the DataFrame you're touching unless this is set to True. It's False by default.


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
