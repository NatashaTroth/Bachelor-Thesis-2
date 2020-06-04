import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import sys
from sklearn.manifold import TSNE
import ipyvolume as ipv
from clustering import calculate_PCA
import json

# TODO: index data https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html


class DataFile:
    def __init__(self, directoryPath):
        self.df = self.create_data_frame(directoryPath)

    def create_data_frame(self, directory_path):
        print("in createDataFrame")
        pathlist = Path(directory_path).glob('**/*.csv')
        list_of_files = []
        for path in pathlist:
            list_of_files.append(str(path))
        print(list_of_files)
        if len(list_of_files) > 1:
            return pd.concat(map(pd.read_csv, list_of_files), ignore_index=True)
        if len(list_of_files) == 1:
            return pd.read_csv(list_of_files[0])
        raise Exception(
            "No *.csv files found in the directory " + directory_path)
        # return df

    def print_file(self):
        print(self.df)

    # def saveFile(self):
    #     print("saving file...")
    #     self.df.to_csv(
    #         '/Users/natashatroth/Documents/FHS/6Semester/Bac2/testData/cleanedFile.csv', index=False)

    def plot(self):
        self.df.plot(kind='bar')

    def clean_data(self, number_same_attributes):
        time_column = self.df['TIME']
        self.df = self.df.drop(columns=['TIME'])
        self.compress_same_attribute_columns(number_same_attributes)
        # self.remove_columns_with_many_empty_values(30, numberSameAttributes)
        self.remove_rows_with_wrong_values()
        self.normalize_columns()

    def replace_non_int_with_NaN(self):
        print("replacing non int with NaN...")
        for column in self.df:
            if(not column.startswith("TIME")):
                self.replace_non_int_with_NaN_per_column(column)

    def remove_rows_with_wrong_values(self):
        print("removing rows with wrong values...")
        self.df.dropna(inplace=True)

    def replace_non_int_with_NaN_per_column(self, column):
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

    def remove_empty_rows(self):
        print("in remove empty rows")
        self.df.dropna(axis=0, how='all',  inplace=True)

    def remove_columns_with_many_empty_values(self, threshold, number_same_attributes):
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
            if(count == number_same_attributes):
                average /= number_same_attributes
                if(100/row_count*average >= threshold):
                    for i in range(1, number_same_attributes + 1):
                        self.df.drop(index[:-1] + str(i), axis=1, inplace=True)
                count = 0
                average = 0

    def normalize_columns(self):
        print("normalizing columns...")
        # timeColumn = self.df['TIME']
        # self.df = self.df.drop(columns=['TIME'])
        scaler = MinMaxScaler()
        self.df[self.df.columns] = scaler.fit_transform(
            self.df[self.df.columns])
        # self.df['TIME'] = timeColumn

    def compress_same_attribute_columns(self, numberSameAttributes):
        # startIndex, EndIndex
        # print(list(self.df.columns))
        self.print_file()

        # columnNames = ['ACC1', 'ACC2', 'ACC3', 'ACC4', 'AUDIO1', 'AUDIO2', 'AUDIO3', 'AUDIO4', 'SCRN1', 'SCRN2', 'SCRN3', 'SCRN4', 'NOTIF1', 'NOTIF2', 'NOTIF3', 'NOTIF4', 'LIGHT1', 'LIGHT2',
        #                'LIGHT3', 'LIGHT4', 'APP_VID1', 'APP_VID2', 'APP_VID3', 'APP_VID4', 'APP_COMM1', 'APP_COMM2', 'APP_COMM3', 'APP_COMM4', 'APP_OTHER1', 'APP_OTHER2', 'APP_OTHER3', 'APP_OTHER4']

        # columnNames2 = ['ACC1', 'ACC2', 'ACC3', 'ACC4', 'ACC5', 'ACC6', 'AUDIO1', 'AUDIO2', 'AUDIO3', 'AUDIO4', 'AUDIO5', 'AUDIO6', 'SCRN1', 'SCRN2', 'SCRN3', 'SCRN4', 'SCRN5', 'SCRN6', 'NOTIF1', 'NOTIF2', 'NOTIF3', 'NOTIF4', 'NOTIF5', 'NOTIF6', 'LIGHT1', 'LIGHT2', 'LIGHT3',
        #                 'LIGHT4', 'LIGHT5', 'LIGHT6', 'APP_VID1', 'APP_VID2', 'APP_VID3', 'APP_VID4', 'APP_VID5', 'APP_VID6', 'APP_COMM1', 'APP_COMM2', 'APP_COMM3', 'APP_COMM4', 'APP_COMM5', 'APP_COMM6', 'APP_OTHER1', 'APP_OTHER2', 'APP_OTHER3', 'APP_OTHER4', 'APP_OTHER5', 'APP_OTHER6']
        distinctAttributes = ['ACC', 'AUDIO', 'SCRN', 'NOTIF',
                              'LIGHT', 'APP_VID',  'APP_COMM',  'APP_OTHER']
        newDf = pd.DataFrame()

        for attribute in distinctAttributes:
            if self.contains_column(attribute):
                start_column = attribute + "1"
                end_column = attribute + "" + str(numberSameAttributes)
                col = self.df.loc[:, start_column:end_column]
                newDf[attribute] = col.mean(axis=1)

        self.df = newDf

    def contains_column(self, attribute):
        if len(self.df.columns[self.df.columns.str.contains(pat=attribute)]) > 0:
            return True
        return False
