import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import sys
from sklearn.manifold import TSNE
import ipyvolume as ipv
from clustering import calculatePCA
import json

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

    def cleanData(self, numberSameAttributes):
        timeColumn = self.df['TIME']
        self.df = self.df.drop(columns=['TIME'])
        # self.removeColumnsWithManyEmptyValues(30, numberSameAttributes)
        self.removeRowsWithWrongValues()
        self.normalizeColumns()

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

    def removeColumnsWithManyEmptyValues(self, threshold, numberSameAttributes):
        print("removing columns with many empty values...")
        print(self.df.isnull().sum())
        # remove columns where percent of rows with empty is above threshold
        # row count ... 100%
        # no of null ... x

        row_count = self.df.shape[0]
        nulls_count = self.df.isnull().sum().to_frame()
        count = 0
        average = 0
        for index, row in nulls_count.iterrows():
            average += row[0]
            count += 1
            if(count == numberSameAttributes):
                average /= numberSameAttributes
                if(100/row_count*average >= threshold):
                    for i in range(1, numberSameAttributes + 1):
                        self.df.drop(index[:-1] + str(i), axis=1, inplace=True)
                count = 0
                average = 0

    def normalizeColumns(self):
        print("normalizing columns...")
        # timeColumn = self.df['TIME']
        # self.df = self.df.drop(columns=['TIME'])
        scaler = MinMaxScaler()
        self.df[self.df.columns] = scaler.fit_transform(
            self.df[self.df.columns])
        # self.df['TIME'] = timeColumn

    def compressSameAttributeColumns(self, numberSameAttributes):
        # startIndex, EndIndex
        # print(list(self.df.columns))
        self.printFile()

        # columnNames = ['ACC1', 'ACC2', 'ACC3', 'ACC4', 'AUDIO1', 'AUDIO2', 'AUDIO3', 'AUDIO4', 'SCRN1', 'SCRN2', 'SCRN3', 'SCRN4', 'NOTIF1', 'NOTIF2', 'NOTIF3', 'NOTIF4', 'LIGHT1', 'LIGHT2',
        #                'LIGHT3', 'LIGHT4', 'APP_VID1', 'APP_VID2', 'APP_VID3', 'APP_VID4', 'APP_COMM1', 'APP_COMM2', 'APP_COMM3', 'APP_COMM4', 'APP_OTHER1', 'APP_OTHER2', 'APP_OTHER3', 'APP_OTHER4']

        # columnNames2 = ['ACC1', 'ACC2', 'ACC3', 'ACC4', 'ACC5', 'ACC6', 'AUDIO1', 'AUDIO2', 'AUDIO3', 'AUDIO4', 'AUDIO5', 'AUDIO6', 'SCRN1', 'SCRN2', 'SCRN3', 'SCRN4', 'SCRN5', 'SCRN6', 'NOTIF1', 'NOTIF2', 'NOTIF3', 'NOTIF4', 'NOTIF5', 'NOTIF6', 'LIGHT1', 'LIGHT2', 'LIGHT3',
        #                 'LIGHT4', 'LIGHT5', 'LIGHT6', 'APP_VID1', 'APP_VID2', 'APP_VID3', 'APP_VID4', 'APP_VID5', 'APP_VID6', 'APP_COMM1', 'APP_COMM2', 'APP_COMM3', 'APP_COMM4', 'APP_COMM5', 'APP_COMM6', 'APP_OTHER1', 'APP_OTHER2', 'APP_OTHER3', 'APP_OTHER4', 'APP_OTHER5', 'APP_OTHER6']
        distinctAttributes = ['ACC', 'AUDIO', 'SCRN', 'NOTIF',
                              'LIGHT', 'APP_VID',  'APP_COMM',  'APP_OTHER']
        newDf = pd.DataFrame()

        for attribute in distinctAttributes:
            if self.containsColumn(attribute):
                startColumn = attribute + "1"
                endColumn = attribute + "" + str(numberSameAttributes)
                col = self.df.loc[:, startColumn:endColumn]
                newDf[attribute] = col.mean(axis=1)

        self.df = newDf

    def containsColumn(self, attribute):
        if len(self.df.columns[self.df.columns.str.contains(pat=attribute)]) > 0:
            return True
        return False
