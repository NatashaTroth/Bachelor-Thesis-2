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
        """ read .csv files from a directory, add a color to each user and concatenate to a pandas dataFrame

            Parameters
            ----------
            directory_path : str
                The directory path where the .csv files are located

            Returns
            -------
            dataFrame object
        """
        pathlist = Path(directory_path).glob('**/*.csv')
        list_of_files = []
        list_of_dataframes = []
        # test = 0
        for path in pathlist:
            list_of_files.append(str(path))
            # if test >= 10:
            #     break
            # test += 1
        if len(list_of_files) > 1:
            return pd.concat(map(self.read_csv_file_add_color, list_of_files), ignore_index=True)
        if len(list_of_files) == 1:
            return pd.read_csv(list_of_files[0])
        raise Exception(
            "No *.csv files found in the directory " + directory_path)

    def read_csv_file_add_color(self, path):
        """ read .csv file and add a color (1 color, per file, per test subject)

            Parameters
            ----------
            path : str
                .csv file path

            Returns
            -------
            dataFrame object
        """
        df = pd.read_csv(path)
        # add color to each user to tell which data points belong to the same user (to tell where the chain was coming from)
        random.seed(self.color_random_int)
        self.color_random_int += 1
        random_color = "#" + "%06x" % random.randint(0, 0xFFFFFF)
        df["COLOR"] = random_color
        return df

    def print_file(self):
        """ prints the dataFrame object in the console"""
        print(self.df)

    def clean_data(self, number_columns_to_use=1):
        """ executes multiple functions to clean the raw data in the dataFrame in-place

            Parameters
            ----------
            number_columns_to_use : int, optional
                number of columns (1-N) to average and calculate with (default 1)
        """
        print("cleaning data...")
        self.remove_columns(["TIME"])
        # self.df = self.df.iloc[::3, :]
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
        """ removes dataFrame rows with missing values or null values"""
        print("  removing rows with wrong values...")
        self.df.dropna(inplace=True)

    def extract_rows_with_percent_non_zero_values(self, percent):
        """ removes rows with not enough non-zero values
            e.g. percent = 75: Keep rows with at least 75% values that aren't 0. Removes rows with more than 25% cells with 0. 

            Parameters
            ----------
            percent : int
                percent of non-zero values
        """
        self.df = self.df[((self.df != 0).sum(axis=1) - 1) /
                          (len(self.df.columns) - 1) >= percent/100]

    def remove_columns_with_many_empty_values(self, threshold, number_columns_to_use):
        """ remove columns, where the number of values exceeds a given threshold
            Parameters
            ----------
            threshold : int
                remove columns where number of empty fields is above it
            number_columns_to_use : int
                number of columns (1-N) to average and calculate with (default 1)
        """

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
            if(count == number_columns_to_use):
                average /= number_columns_to_use
                if(100/row_count*average >= threshold):
                    for i in range(1, number_columns_to_use + 1):
                        self.df.drop(index[:-1] + str(i), axis=1, inplace=True)
                count = 0
                average = 0

    def extract_columns(self, number_columns_to_use):
        """ extract the given number of columns to use (1-N)
            e.g. number_columns_to_use = 2: extract columns 1 and 2 from every feature (e.g. ACC1 and ACC2)

            Parameters
            ----------
            number_columns_to_use : int
                number of columns (1-N) to average and calculate with (default 1)
        """
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
        """ remove columns whose names are in the column_names parameter

            Parameters
            ----------
            column_names : list
                list (array) of column names
        """
        for column_name in column_names:
            self.df = self.df.drop(self.df.filter(
                regex=column_name).columns, axis=1)

    def compress_same_attribute_columns(self, number_columns_to_use):
        """ average multiple columns of the same feature (e.g. ACC1-ACC4) and compress into one column

            Parameters
            ----------
            number_columns_to_use : int
                number of columns (1-N) to average and calculate with (default 1)
        """
        print("  compressing same attribute columns...")
        new_df = pd.DataFrame()

        for attribute in self.distinctAttributes:
            if self.contains_column(attribute):
                start_column = attribute + "1"
                end_column = attribute + "" + str(number_columns_to_use)
                col = self.df.loc[:, start_column:end_column]
                new_df[attribute] = col.mean(axis=1)

        self.df = new_df

    def contains_column(self, attribute):
        """ checks if dataFrame contains a column, whose name contains the given attribute str

            Parameters
            ----------
            attribute : str
                name to check if a column contains it (e.g. ACC)
        """
        if len(self.df.columns[self.df.columns.str.contains(pat=attribute)]) > 0:
            return True
        return False

    def normalize_columns(self):
        """ normalize the values in the dataFrame using z-score normalization"""
        print("  normalizing columns...")
        scaler = StandardScaler()
        self.df[self.df.columns] = scaler.fit_transform(
            self.df[self.df.columns])

    def calculate_PCA(self, number_components, graphs):
        """ apply PCA dimensionality reduction to the dataFrame

            Parameters
            ----------
            number_components : int
                number of components to return
            graphs : Boolean
                render PCA scatter plots and component bar chart graphs (True), or not (False)
        """
        self.pca = calculate_PCA(
            self.df, number_components, graphs, self.colors)

    def calculate_TSNE(self, number_components, graphs):
        """ apply t-SNE dimensionality reduction to the dataFrame

            Parameters
            ----------
            number_components : int
                number of components to return
            graphs : Boolean
                render t-SNE scatter plot graphs (True), or not (False)
        """
        self.tsne = calculate_TSNE(
            self.df, number_components, graphs, self.colors)

    def spectral_clustering(self, data_type, graphs=False):
        """ apply spectral clustering to the dataFrame
            the resulting scores are saved into the self.spectral_scores property

            Parameters
            ----------
            data_type : str
                type of data from the dataFrame to feed into the clustering algorithm (options: "PCA", "t-SNE", or "" which means the entire dataFrame)
            graphs : Boolean
                render clustering scatter plot graphs (True), or not (False)
        """
        if data_type == 'PCA':
            self.spectral_scores = spectral_clustering(
                self.pca, data_type, graphs)
        if data_type == 'TSNE':
            self.spectral_scores = spectral_clustering(
                self.tsne, data_type, graphs)

    def dbscan_clustering(self, data_type, graphs=False):
        """ apply DBSCAN clustering to the dataFrame
            the resulting scores are saved into the self.dbscan_scores property

            Parameters
            ----------
            data_type : str
                type of data from the dataFrame to feed into the clustering algorithm (options: "PCA", "t-SNE", or "" which means the entire dataFrame)
            graphs : Boolean
                render clustering scatter plot graphs (True), or not (False)
        """
        if data_type == 'PCA':
            # predict_eps_dbscan_parameter(self.pca)
            self.dbscan_scores = dbscan_clustering(self.pca, data_type, graphs)
        elif data_type == 'TSNE':
            # predict_eps_dbscan_parameter(self.tsne)
            self.dbscan_scores = dbscan_clustering(
                self.tsne, data_type, graphs)
        else:
            # predict_eps_dbscan_parameter(self.df)
            self.dbscan_scores = dbscan_clustering(self.df)

    def optics_clustering(self, data_type, graphs=False):
        """ apply OPTICS clustering to the dataFrame
            the resulting scores are saved into the self.optics_scores property

            Parameters
            ----------
            data_type : str
                type of data from the dataFrame to feed into the clustering algorithm (options: "PCA", "t-SNE", or "" which means the entire dataFrame)
            graphs : Boolean
                render clustering scatter plot and reachability plot graphs (True), or not (False)
        """
        if data_type == 'PCA':
            self.optics_scores = optics_clustering(self.pca, data_type, graphs)
        elif data_type == 'TSNE':
            self.optics_scores = optics_clustering(
                self.tsne, data_type, graphs)
        else:
            self.optics_scores = optics_clustering(self.df)

    def agglomerative_clustering(self, data_type, graphs=False):
        """ apply agglomerative clustering to the dataFrame
            the resulting scores are saved into the self.agglomerative_scores property

            Parameters
            ----------
            data_type : str
                type of data from the dataFrame to feed into the clustering algorithm (options: "PCA", "t-SNE", or "" which means the entire dataFrame)
            graphs : Boolean
                render clustering scatter plot graphs (True), or not (False)
        """
        if data_type == 'PCA':
            self.agglomerative_scores = agglomerative_clustering(
                self.pca, data_type, graphs)
        elif data_type == 'TSNE':
            self.agglomerative_scores = agglomerative_clustering(
                self.tsne, data_type, graphs)
        else:
            self.agglomerative_scores = agglomerative_clustering(self.df)
