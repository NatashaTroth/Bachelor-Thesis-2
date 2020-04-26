import pandas as pd
import numpy as np

# TODO: index data https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html


class DataFile:
    def __init__(self, filePath):
        self.df = df = pd.read_csv(filePath)
        self.filePath = filePath

    def replaceColumnNonIntWithNaN(self, column):
        counter = 0
        # print("COLOUMN:" + column)
        for row in self.df[column]:
            # print(row)
            try:
                # test if value can be converted to an int - if not, it is not a number
                int(row)
                pass
            except ValueError:
                self.df.loc[counter, column] = np.nan
            counter += 1

    def replaceNonIntWithNaN(self):
        # print("FILE:" + self.filePath)
        for column in self.df:
            self.replaceColumnNonIntWithNaN(column)

# print df.isnull().values.any()
