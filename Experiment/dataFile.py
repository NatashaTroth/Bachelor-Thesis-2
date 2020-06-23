import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.manifold import TSNE
from dimensionalityReduction import calculate_PCA, calculate_TSNE
from clustering import spectral_clustering, dbscan_clustering, optics_clustering, agglomerative_clustering, predict_eps_dbscan_parameter
import random


class DataFile:
    def __init__(self, directoryPath):
        self.color_random_int = 0
        self.df = self.create_data_frame(directoryPath)
        self.distinctAttributes = ['ACC', 'AUDIO', 'SCRN', 'NOTIF',
                                   'LIGHT', 'APP_VID',  'APP_COMM',  'APP_OTHER']

    def create_data_frame(self, directory_path):
        pathlist = Path(directory_path).glob('**/*.csv')
        list_of_files = []
        list_of_dataframes = []
        test = 0
        for path in pathlist:
            list_of_files.append(str(path))
        if len(list_of_files) > 1:
            return pd.concat(map(self.read_csv_file_add_color, list_of_files), ignore_index=True)
        if len(list_of_files) == 1:
            return pd.read_csv(list_of_files[0])
        raise Exception(
            "No *.csv files found in the directory " + directory_path)

    def read_csv_file_add_color(self, path):
        df = pd.read_csv(path)
        # add color to each user to tell which data points belong to the same user (to tell where the chain was coming from)
        random.seed(self.color_random_int)
        self.color_random_int += 1
        random_color = "#" + "%06x" % random.randint(0, 0xFFFFFF)
        df["COLOR"] = random_color
        return df

    def print_file(self):
        print(self.df)

    def plot(self):
        self.df.plot(kind='bar')

    def clean_data(self, number_columns_to_use=1):
        print("cleaning data...")
        self.remove_columns(["TIME"])
        # self.remove_columns_with_many_empty_values(30, number_columns_to_use)
        self.remove_rows_with_wrong_values()
        self.colors = self.df["COLOR"].to_numpy()
        self.remove_columns(["COLOR"])
        self.extract_columns(number_columns_to_use)

        # if remove rows, also need to adjust the colors for the tester subjects
        self.df["COLOR"] = self.colors
        self.extract_rows_with_percent_non_zero_values(50)
        self.colors = self.df["COLOR"].to_numpy()
        self.remove_columns(["COLOR"])
        if number_columns_to_use > 1:
            self.compress_same_attribute_columns(number_columns_to_use)
        self.normalize_columns()

    def remove_rows_with_wrong_values(self):
        print("  removing rows with wrong values...")
        self.df.dropna(inplace=True)

    def extract_rows_with_percent_non_zero_values(self, percent):
        # e.g. percent = 60: removes rows with 60 percent or more 0s (where less than 40% other values that != 0)
        self.df = self.df[((self.df != 0).sum(axis=1) - 1) /
                          (len(self.df.columns) - 1) >= percent/100]

    def remove_columns_with_many_empty_values(self, threshold, number_same_attributes):
        print("  removing columns with many empty values...")
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
            self.df = self.df.drop(self.df.filter(
                regex=column_name).columns, axis=1)

    def compress_same_attribute_columns(self, numberSameAttributes):
        print("  compressing same attribute columns...")
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
        scaler = StandardScaler()
        self.df[self.df.columns] = scaler.fit_transform(
            self.df[self.df.columns])

    def calculate_PCA(self, number_components, graphs):
        self.pca = calculate_PCA(
            self.df, number_components, graphs, self.colors)

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
