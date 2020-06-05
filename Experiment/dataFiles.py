import pandas as pd
import numpy as np
from dataFile import DataFile
from pathlib import Path
import sys


class DataFiles:
    def __init__(self, directoryPath):
        self.dataFiles = self.fetchDataFiles(directoryPath)

    def fetchDataFiles(self, directoryPath):
        # print("here 1")
        pathlist = Path(directoryPath).glob('**/*.csv')
        listOfFiles = []
        for path in pathlist:
            listOfFiles.append(DataFile(str(path)))
        return listOfFiles

    def printFiles(self):
        for dataFile in self.dataFiles:
            dataFile.printFile()

    def cleanData(self):
        # try:
        for dataFile in self.dataFiles:
            dataFile.replaceNonIntWithNaN()
            dataFile.removeRowsWithWrongValues()
            # dataFile.plot()
            dataFile.normalizeColumns()

        # except:
        #     print(
        #         "Something went wrong in the cleanData function. ", sys.exc_info()[0])