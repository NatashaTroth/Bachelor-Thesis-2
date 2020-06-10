import pandas as pd
import numpy as np
from sklearn import preprocessing
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys
from sklearn.manifold import TSNE
import ipyvolume as ipv
import json
from dimensionalityReduction import calculate_PCA
from dimensionalityReduction import calculate_TSNE
from clustering import spectral_clustering
from clustering import dbscan_clustering
from clustering import optics_clustering
from clustering import agglomerative_clustering


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
        if len(list_of_files) > 1:
            return pd.concat(map(pd.read_csv, list_of_files), ignore_index=True)
        if len(list_of_files) == 1:
            return pd.read_csv(list_of_files[0])
        raise Exception(
            "No *.csv files found in the directory " + directory_path)
        # return df

    def print_file(self):
        print(self.df)

    def plot(self):
        self.df.plot(kind='bar')

    def clean_data(self, number_same_attributes):
        print("cleaning data...")
        # time_column = self.df['TIME']
        self.df = self.df.drop(columns=['TIME'])
        # self.remove_columns_with_many_empty_values(30, number_same_attributes)
        self.remove_rows_with_wrong_values()
        # self.compress_same_attribute_columns(number_same_attributes)
        self.normalize_columns()
        # print(self.df)
        # print("MAX VALUES...")
        # print(self.df.max(axis=0))
        # print("\nMIN VALUES...")
        # print(self.df.min(axis=0))

    def remove_rows_with_wrong_values(self):
        print("  removing rows with wrong values...")
        self.df.dropna(inplace=True)

    def remove_columns_with_many_empty_values(self, threshold, number_same_attributes):
        print("  removing columns with many empty values...")
        # print(self.df.isnull().sum())
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
        print("  normalizing columns...")
        # scaler = MinMaxScaler()
        scaler = StandardScaler()
        self.df[self.df.columns] = scaler.fit_transform(
            self.df[self.df.columns])

    def compress_same_attribute_columns(self, numberSameAttributes):
        print("  compressing same attribute columns...")
        # self.print_file()
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

    def calculate_PCA(self, number_components, graphs):
        self.pca = calculate_PCA(self.df, number_components, graphs)

    def calculate_TSNE(self, number_components, graphs):
        self.tsne = calculate_TSNE(self.df, number_components, graphs)

    def spectral_clustering(self, dataType, graphs=False):
        if dataType == 'PCA':
            spectral_clustering(self.pca, dataType, graphs)
        if dataType == 'TSNE':
            spectral_clustering(self.tsne, dataType, graphs)

    def dbscan_clustering(self, dataType, graphs=False):
        if dataType == 'PCA':
            dbscan_clustering(self.pca, dataType, graphs)
        elif dataType == 'TSNE':
            dbscan_clustering(self.tsne, dataType, graphs)
        else:
            dbscan_clustering(self.df)

    def optics_clustering(self, dataType, graphs=False):
        if dataType == 'PCA':
            optics_clustering(self.pca, dataType, graphs)
        elif dataType == 'TSNE':
            optics_clustering(self.tsne, dataType, graphs)
        else:
            optics_clustering(self.df)

    def agglomerative_clustering(self, dataType, graphs=False):
        if dataType == 'PCA':
            agglomerative_clustering(self.pca, dataType, graphs)
        elif dataType == 'TSNE':
            agglomerative_clustering(self.tsne, dataType, graphs)
        else:
            agglomerative_clustering(self.df)
