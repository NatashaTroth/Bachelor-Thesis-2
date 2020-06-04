import unittest
from pandas._testing import assert_frame_equal
import pandas as pd
# import numpy as np
from dataFile import DataFile
from dataFiles import DataFiles
from pathlib import Path
import sys
from clustering import calculate_PCA

# Disclaimer: the numbers in testData.csv were created using a random generator (see randomNrs.js) - not from test users!!


class TestStringMethods(unittest.TestCase):

    def test_clean_data(self):

        data_file = DataFile("./Experiment/testData/testNormal")
        # ---remove rows with missing value & time column---
        data_file.df = data_file.df.drop(columns=['TIME'])
        data_file.remove_rows_with_wrong_values()

        result_after_removed_rows_with_null = pd.read_json(
            './Experiment/testData/testNormal/resultAfterRemovedRowsWithNull.json', orient='index')
        assert_frame_equal(
            data_file.df, result_after_removed_rows_with_null,  check_dtype=False)

        # ---compression columns (so ACC1-ACCn becomes just ACC)---
        data_file.compress_same_attribute_columns(4)
        result_after_compressing_columns = pd.read_json(
            './Experiment/testData/testNormal/resultAfterCompressingColumns.json', orient='index')
        assert_frame_equal(
            data_file.df, result_after_compressing_columns,  check_dtype=False)

        # ---normalize data---
        data_file.normalize_columns()
        result_after_normalization = pd.read_json(
            './Experiment/testData/testNormal/resultAfterNormalization.json', orient='index')
        assert_frame_equal(
            data_file.df, result_after_normalization,  check_dtype=False)


if __name__ == '__main__':
    unittest.main()
