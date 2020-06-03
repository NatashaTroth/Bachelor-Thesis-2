import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import sys
from sklearn.manifold import TSNE
import ipyvolume as ipv
from clustering import calculatePCA

# TODO: index data https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html


class DataFile:
    def __init__(self, directoryPath):
        self.df = self.createDataFrame(directoryPath)

    def createDataFrame(self, directoryPath):
        print("in createDataFrame")
        pathlist = Path(directoryPath).glob('**/*.csv')
        listOfFiles = []
        for path in pathlist:
            listOfFiles.append(str(path))
        return pd.concat(
            map(pd.read_csv, listOfFiles), ignore_index=True)
        # return df

    def printFile(self):
        print(self.df)

    # def saveFile(self):
    #     print("saving file...")
    #     self.df.to_csv(
    #         '/Users/natashatroth/Documents/FHS/6Semester/Bac2/testData/cleanedFile.csv', index=False)

    def plot(self):
        self.df.plot(kind='bar')

    def cleanData(self):
        # self.replaceNonIntWithNaN()
        # print(self.df.isnull().sum()) #todo: remove columns with high amount of missing values
        timeColumn = self.df['TIME']
        self.df = self.df.drop(columns=['TIME'])
        self.removeRowsWithWrongValues()
        self.normalizeColumns()
        # calculatePCA(self.df)

    def replaceNonIntWithNaN(self):
        print("replacing non int with NaN...")
        for column in self.df:
            if(not column.startswith("TIME")):
                self.replaceNonIntWithNaNPerColumn(column)

    def removeRowsWithWrongValues(self):
        print("removing rows with wrong values...")
        self.df.dropna(inplace=True)

    def replaceNonIntWithNaNPerColumn(self, column):
        counter = 0

        for row in self.df[column]:
            if(not column.startswith("TIME")):
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

    def normalizeColumns(self):
        print("normalizing columns...")
        # timeColumn = self.df['TIME']
        # self.df = self.df.drop(columns=['TIME'])
        scaler = MinMaxScaler()
        self.df[self.df.columns] = scaler.fit_transform(
            self.df[self.df.columns])
        # self.df['TIME'] = timeColumn
