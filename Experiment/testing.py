import unittest
from pandas._testing import assert_frame_equal
import pandas as pd
# import numpy as np
from dataFile import DataFile
from dataFiles import DataFiles
from pathlib import Path
import sys
from dimensionalityReduction import calculate_PCA
from clusterEvaluation import compare_scores

# Disclaimer: the numbers in the test *.csv files were created using a random generator (see randomNrs.js) - not from real test users!!


class TestStringMethods(unittest.TestCase):

    def test_clean_data(self):

        data_file = DataFile("./testData/testNormal")
        # ---remove rows with missing value & time column---
        data_file.df = data_file.df.drop(columns=['TIME'])
        data_file.remove_rows_with_wrong_values()

        result_after_removed_rows_with_null = pd.read_json(
            './testData/testNormal/resultAfterRemovedRowsWithNull.json', orient='index')
        assert_frame_equal(
            data_file.df, result_after_removed_rows_with_null,  check_dtype=False)

        # ---compression columns (so ACC1-ACCn becomes just ACC)---
        data_file.compress_same_attribute_columns(4)
        result_after_compressing_columns = pd.read_json(
            './testData/testNormal/resultAfterCompressingColumns.json', orient='index')
        assert_frame_equal(
            data_file.df, result_after_compressing_columns,  check_dtype=False)

        # ---normalize data---
        data_file.normalize_columns()
        print(data_file.df)
        result_after_normalization = pd.read_json(
            './testData/testNormal/resultAfterNormalization.json', orient='index')
        assert_frame_equal(
            data_file.df, result_after_normalization,  check_dtype=False)

    def test_extract_colums(self):
        data_file = DataFile("./testData/testNormal")
        data_file.df = data_file.df.drop(columns=['TIME'])

        data_file.remove_rows_with_wrong_values()
        result_after_removed_rows_with_null = pd.read_json(
            './testData/testNormal/resultAfterRemovedRowsWithNull.json', orient='index')
        assert_frame_equal(
            data_file.df, result_after_removed_rows_with_null,  check_dtype=False)

        data_file.extract_columns(2)
        result_after_extract_columns = pd.read_json(
            './testData/testNormal/resultsAfterExtractColumns.json', orient='index')
        assert_frame_equal(
            data_file.df, result_after_extract_columns,  check_dtype=False)

    def test_remove_colums_with_many_missing_values(self):
        data_file = DataFile("./testData/testRemoveColumns")
        # ---remove rows with missing value & time column---
        data_file.df = data_file.df.drop(columns=['TIME'])
        # print(data_file.df.to_json(orient="index"))
        data_file.remove_columns_with_many_empty_values(30, 4)
        result_after_removing_columns = pd.read_json(
            './testData/testRemoveColumns/resultAfterRemovingColumns.json', orient='index')
        assert_frame_equal(
            data_file.df, result_after_removing_columns,  check_dtype=False)

    def test_evaluation_compare_scores(self):
        scores_1h_dbscan = [-0.4465066, 2.3737173437900196, 34.818951039037465]
        scores_3h_dbscan = [-0.2325946, 2.082509570770939, 176.03847255411213]

        result_dbscan = compare_scores(scores_1h_dbscan, scores_3h_dbscan)
        self.assertEqual(result_dbscan, 2)

        scores_1h_optics = [-0.10805944, 1.234722639643406, 18.19778900973744]
        scores_3h_optics = [-0.06738582,
                            1.6176452947156887, 16.683104491741254]

        result_optics = compare_scores(scores_1h_optics, scores_3h_optics)
        self.assertEqual(result_optics, 1)

    def test_remove_rows_with_percent_zero(self):
        data_file = DataFile("./testData/testRemoveRowsWithZero")
        # print(data_file.df.to_json(orient='index'))
        initial_file = pd.read_json(
            './testData/testRemoveRowsWithZero/testData.json', orient='index')
        assert_frame_equal(
            data_file.df, initial_file,  check_dtype=False)

        data_file.remove_rows_with_percent_zero(25)
        result_after_removing_rows_more_25_percent_zero = pd.read_json(
            './testData/testRemoveRowsWithZero/testRemove25Percent.json', orient='index')
        assert_frame_equal(
            data_file.df, result_after_removing_rows_more_25_percent_zero,  check_dtype=False)

        data_file = DataFile("./testData/testRemoveRowsWithZero")
        data_file.remove_rows_with_percent_zero(50)
        result_after_removing_rows_more_50_percent_zero = pd.read_json(
            './testData/testRemoveRowsWithZero/testRemove50Percent.json', orient='index')
        assert_frame_equal(
            data_file.df, result_after_removing_rows_more_50_percent_zero,  check_dtype=False)

        data_file = DataFile("./testData/testRemoveRowsWithZero")
        data_file.remove_rows_with_percent_zero(49)
        print(data_file.df)
        result_after_removing_rows_more_49_percent_zero = pd.read_json(
            './testData/testRemoveRowsWithZero/testRemove49Percent.json', orient='index')
        assert_frame_equal(
            data_file.df, result_after_removing_rows_more_49_percent_zero,  check_dtype=False)

        data_file = DataFile("./testData/testRemoveRowsWithZero")
        data_file.remove_rows_with_percent_zero(75)
        print(data_file.df)
        result_after_removing_rows_more_75_percent_zero = pd.read_json(
            './testData/testRemoveRowsWithZero/testRemove75Percent.json', orient='index')
        assert_frame_equal(
            data_file.df, result_after_removing_rows_more_75_percent_zero,  check_dtype=False)


if __name__ == '__main__':
    unittest.main()
