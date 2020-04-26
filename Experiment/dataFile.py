import pandas as pd
import numpy as np

# TODO: index data https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html


class DataFile:
    def __init__(self, filePath):
        self.df = df = pd.read_csv(filePath)

    def replaceColumnNotIntWithNaN(self, column):
        counter = 0
        print(column)
        for row in self.df[column]:
            # print(row)

            try:
                int(row)
                pass
            except ValueError:
                self.df.loc[counter, column] = np.nan
            counter += 1

    def replaceAllNotIntWithNaN(self):
        for column in self.df:
            print(column)
            self.replaceColumnNotIntWithNaN(column)

# print df.isnull().values.any()
