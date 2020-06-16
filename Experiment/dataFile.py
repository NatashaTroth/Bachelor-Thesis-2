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
from dimensionalityReduction import calculate_PCA, calculate_TSNE
from clustering import spectral_clustering, dbscan_clustering, optics_clustering, agglomerative_clustering, predict_eps_dbscan_parameter
import random

# TODO: index data https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html


class DataFile:
    def __init__(self, directoryPath):
        self.df = self.create_data_frame(directoryPath)
        self.distinctAttributes = ['ACC', 'AUDIO', 'SCRN', 'NOTIF',
                                   'LIGHT', 'APP_VID',  'APP_COMM',  'APP_OTHER']

    def create_data_frame(self, directory_path):
        print("in createDataFrame")
        pathlist = Path(directory_path).glob('**/*.csv')
        list_of_files = []
        list_of_dataframes = []
        test = 0
        for path in pathlist:
            list_of_files.append(str(path))

            test += 1
            if test == 25:
                break
        #     df = pd.read_csv(str(path), index_col=None, header=0)
        #     li.append(df)

        # frame = pd.concat(li, axis=0, ignore_index=True)

        # color = "%06x" % random.randint(0, 0xFFFFFF)
        print(list_of_files)
        if len(list_of_files) > 1:
            return pd.concat(map(self.read_csv_file_add_color, list_of_files), ignore_index=True)
        if len(list_of_files) == 1:
            return pd.read_csv(list_of_files[0])
        raise Exception(
            "No *.csv files found in the directory " + directory_path)
        # return df

    def read_csv_file_add_color(self, path):
        df = pd.read_csv(path)
        # print("%06x" % random.randint(0, 0xFFFFFF))
        random_color = "#" + "%06x" % random.randint(0, 0xFFFFFF)
        print(random_color)
        df["COLOR"] = random_color
        return df

    def print_file(self):
        print(self.df)

    def plot(self):
        self.df.plot(kind='bar')

    def clean_data(self, number_columns_to_use=1):
        print("cleaning data...")
        # time_column = self.df['TIME']
        # self.df = self.df.drop(columns=['TIME'])
        print(self.df)
        # self.df = pd.DataFrame([np.arange(len(self.df)) % 2 == 1])
        # self.df = self.df.iloc[::2, :]
        # self.df = self.df.iloc[::3, :]
        # print(self.df)
        print(self.df)

        self.remove_columns(["TIME"])

        # self.remove_columns(["TIME", "AUDIO", "SCRN", "LIGHT", "NOTIF", "APP"])
        # self.remove_columns(["TIME", "APP", "SCRN", "LIGHT", "NOTIF", "AUDIO"])
        # print("----------here---------------")
        # print(self.df)
        # self.remove_columns_with_many_empty_values(30, number_columns_to_use)
        self.remove_rows_with_wrong_values()

        self.colors = self.df["COLOR"].to_numpy()
        print(self.colors)
        self.remove_columns(["COLOR"])

        self.extract_columns(number_columns_to_use)
        # print(self.df)
        if number_columns_to_use > 1:
            self.compress_same_attribute_columns(number_columns_to_use)
        # print(self.df)

        # print(self.df)
        # print("MAX VALUES...")
        # print(self.df.max(axis=0))
        # print("\nMIN VALUES...")
        # print(self.df.min(axis=0))
        # print("---------------HERE.----------------")
        # print(self.df)

        self.normalize_columns()
        # self.df = self.df.iloc[::2]
        # self.df = self.df.iloc[::1]
        # print(self.df)

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

    def extract_columns(self, number_columns_to_use):
        print("  extracting " + str(number_columns_to_use) + " column(s)...")
        new_df = pd.DataFrame()
        counter = 1
        for attribute in self.distinctAttributes:
            if self.contains_column(attribute):
                while counter <= number_columns_to_use:
                    new_df[attribute + str(counter)
                           ] = self.df[attribute + str(counter)]
                    counter += 1

            counter = 1
        self.df = new_df

    def remove_columns(self, column_names):
        for column_name in column_names:
            print(column_name)
            self.df = self.df.drop(self.df.filter(
                regex=column_name).columns, axis=1)
        # self.df = self.df.drop(columns=column_names)

    def compress_same_attribute_columns(self, numberSameAttributes):
        print("  compressing same attribute columns...")
        # self.print_file()
        # distinctAttributes = ['ACC', 'AUDIO', 'SCRN', 'NOTIF',
        #                       'LIGHT', 'APP_VID',  'APP_COMM',  'APP_OTHER']
        new_df = pd.DataFrame()

        for attribute in self.distinctAttributes:
            if self.contains_column(attribute):
                start_column = attribute + "1"
                end_column = attribute + "" + str(numberSameAttributes)
                col = self.df.loc[:, start_column:end_column]
                new_df[attribute] = col.mean(axis=1)

        self.df = new_df

    def contains_column(self, attribute):
        if len(self.df.columns[self.df.columns.str.contains(pat=attribute)]) > 0:
            return True
        return False

    def normalize_columns(self):
        print("  normalizing columns...")
        # scaler = MinMaxScaler()
        scaler = StandardScaler()
        self.df[self.df.columns] = scaler.fit_transform(
            self.df[self.df.columns])

    def calculate_PCA(self, number_components, graphs):
        self.pca = calculate_PCA(self.df, number_components, graphs)

    def calculate_TSNE(self, number_components, graphs):
        self.tsne = calculate_TSNE(
            self.df, number_components, graphs, self.colors)

    def spectral_clustering(self, dataType, graphs=False):
        if dataType == 'PCA':
            self.spectral_scores = spectral_clustering(
                self.pca, dataType, graphs)
        if dataType == 'TSNE':
            self.spectral_scores = spectral_clustering(
                self.tsne, dataType, graphs)

    def dbscan_clustering(self, dataType, graphs=False):
        if dataType == 'PCA':
            # predict_eps_dbscan_parameter(self.pca)
            self.dbscan_scores = dbscan_clustering(self.pca, dataType, graphs)
        elif dataType == 'TSNE':
            # predict_eps_dbscan_parameter(self.tsne)
            self.dbscan_scores = dbscan_clustering(self.tsne, dataType, graphs)
        else:
            # predict_eps_dbscan_parameter(self.df)
            self.dbscan_scores = dbscan_clustering(self.df)

    def optics_clustering(self, dataType, graphs=False):
        if dataType == 'PCA':
            self.optics_scores = optics_clustering(self.pca, dataType, graphs)
        elif dataType == 'TSNE':
            self.optics_scores = optics_clustering(self.tsne, dataType, graphs)
        else:
            self.optics_scores = optics_clustering(self.df)

    def agglomerative_clustering(self, dataType, graphs=False):
        if dataType == 'PCA':
            self.agglomerative_scores = agglomerative_clustering(
                self.pca, dataType, graphs)
        elif dataType == 'TSNE':
            self.agglomerative_scores = agglomerative_clustering(
                self.tsne, dataType, graphs)
        else:
            self.agglomerative_scores = agglomerative_clustering(self.df)
