import unittest
from pandas._testing import assert_frame_equal
import pandas as pd
from dataFile import DataFile

# IMPORTANT: the numbers in the test *.csv files were created using a random number generator (see randomNrs.js) - not from real test users!!


class TestStringMethods(unittest.TestCase):

    def test_clean_data(self):
        data_file = DataFile("./mockTestData/testNormal")
        # ---remove rows with missing value & time column---
        data_file.df = data_file.df.drop(columns=['TIME'])
        data_file.remove_rows_with_wrong_values()

        result_after_removed_rows_with_null = pd.read_json(
            './mockTestData/testNormal/resultAfterRemovedRowsWithNull.json', orient='index')
        assert_frame_equal(
            data_file.df, result_after_removed_rows_with_null,  check_dtype=False)

        # ---compression columns (so ACC1-ACCn becomes just ACC)---
        data_file.compress_same_attribute_columns(4)
        result_after_compressing_columns = pd.read_json(
            './mockTestData/testNormal/resultAfterCompressingColumns.json', orient='index')
        assert_frame_equal(
            data_file.df, result_after_compressing_columns,  check_dtype=False)

        # ---normalize data---
        data_file.normalize_columns()
        result_after_normalization = pd.read_json(
            './mockTestData/testNormal/resultAfterNormalization.json', orient='index')
        assert_frame_equal(
            data_file.df, result_after_normalization,  check_dtype=False)

    def test_extract_colums(self):
        data_file = DataFile("./mockTestData/testNormal")
        data_file.df = data_file.df.drop(columns=['TIME'])

        data_file.remove_rows_with_wrong_values()
        result_after_removed_rows_with_null = pd.read_json(
            './mockTestData/testNormal/resultAfterRemovedRowsWithNull.json', orient='index')
        assert_frame_equal(
            data_file.df, result_after_removed_rows_with_null,  check_dtype=False)

        data_file.extract_columns(2)
        result_after_extract_columns = pd.read_json(
            './mockTestData/testNormal/resultsAfterExtractColumns.json', orient='index')
        assert_frame_equal(
            data_file.df, result_after_extract_columns,  check_dtype=False)

    def test_extract_rows_with_percent_non_zero_values(self):

        colors = ["#ffffff", "#ffffff",
                  "#ffffff", "#ffffff", "#ffffff", "#ffffff"]
        data_file = DataFile("./mockTestData/testRemoveRowsWithZero")
        data_file.df["COLOR"] = colors
        initial_file = pd.read_json(
            './mockTestData/testRemoveRowsWithZero/testData.json', orient='index')
        assert_frame_equal(
            data_file.df, initial_file,  check_dtype=False)

        # remove rows with more than 25% cells with 0. So keep rows with at least 75% values that aren't 0
        data_file.extract_rows_with_percent_non_zero_values(75)
        result_after_removing_rows_more_25_percent_zero = pd.read_json(
            './mockTestData/testRemoveRowsWithZero/testRemove25Percent.json', orient='index')
        assert_frame_equal(
            data_file.df, result_after_removing_rows_more_25_percent_zero,  check_dtype=False)

        # remove rows with more than 50% cells with 0. So keep rows with at least 50% values that aren't 0
        data_file = DataFile("./mockTestData/testRemoveRowsWithZero")
        data_file.df["COLOR"] = colors
        data_file.extract_rows_with_percent_non_zero_values(50)
        result_after_removing_rows_more_50_percent_zero = pd.read_json(
            './mockTestData/testRemoveRowsWithZero/testRemove50Percent.json', orient='index')
        assert_frame_equal(
            data_file.df, result_after_removing_rows_more_50_percent_zero,  check_dtype=False)

        # remove rows with more than 49% cells with 0. So keep rows with at least 51% values that aren't 0
        data_file = DataFile("./mockTestData/testRemoveRowsWithZero")
        data_file.df["COLOR"] = colors
        data_file.extract_rows_with_percent_non_zero_values(51)
        result_after_removing_rows_more_49_percent_zero = pd.read_json(
            './mockTestData/testRemoveRowsWithZero/testRemove49Percent.json', orient='index')
        assert_frame_equal(
            data_file.df, result_after_removing_rows_more_49_percent_zero,  check_dtype=False)

        # remove rows with more than 75% cells with 0. So keep rows with at least 25% values that aren't 0
        data_file = DataFile("./mockTestData/testRemoveRowsWithZero")
        data_file.df["COLOR"] = colors
        data_file.extract_rows_with_percent_non_zero_values(25)
        result_after_removing_rows_more_75_percent_zero = pd.read_json(
            './mockTestData/testRemoveRowsWithZero/testRemove75Percent.json', orient='index')
        assert_frame_equal(
            data_file.df, result_after_removing_rows_more_75_percent_zero,  check_dtype=False)


if __name__ == '__main__':
    unittest.main()
