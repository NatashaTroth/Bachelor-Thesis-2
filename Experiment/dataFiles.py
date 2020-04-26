import pandas as pd
import numpy as np
from dataFile import DataFile
from pathlib import Path


class DataFiles:
    def __init__(self, directoryPath):
        self.dataFiles = self.fetchDataFiles(directoryPath)

    def fetchDataFiles(self, directoryPath):
        pathlist = Path(directoryPath).glob('**/*.csv')
        listOfFiles = []
        for path in pathlist:
            listOfFiles.append(DataFile(str(path)))
        return listOfFiles

    def replaceNonIntWithNaN(self):
        for dataFile in self.dataFiles:
            dataFile.replaceNonIntWithNaN()
