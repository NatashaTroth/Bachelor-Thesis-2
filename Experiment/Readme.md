# Identifying the Ideal Length of Time to Aggregate Smartphone Data, in Order to Obtain Distinct Clusters to Predict Eating Crises

Bachelor Thesis 2, FH Salzburg, MultiMediaTechnology
Salzburg, Austria, 29.06.2020

## Author: 
Natasha Troth

## Advisor: 
FH-Prof. DI Dr. Simon Ginzinger, MSc

# Files
 - [main.py](./main.py): start programme, control what data goes in, what data is printed, which functions (e.g. dimensionality reduction, clustering) are used, how many dimensions, how many columns (time lengths). The file names for the 1h dataset directory and for the 3h dataset directory, can be added as commandline arguments in the respective order (e.g. python main.py "/directoryFiles/1h" "/directoryFiles/3h")
 - [dataFile.py](./dataFile.py): object that manages the data sets (read and concatenate .csv files, clean data, apply dimensionality reductions and clusterings)
 - [dimensionalityReduction.py](./dimensionalityReduction.py): apply and plot t-SNE and PCA
 - [clustering.py](./clustering.py): apply and plot DBSCAN and OPTICS. Also predict DBSCAN Eps parameter (plot k-dist graphs).
 - [clusterEvaluation.py](./clusterEvaluation.py): Calculate Silhouette Score, Davies Bouldin Score, and Caliński Harabasz Score
 - [plot.py](./plot.py): Create 2d and 3d plots (Matplotlib)
 - [testing.py](./testing.py): Unit tests (the numbers in the test *.csv files were created using a random number generator (see randomNrs.js) - not from real test subjects)


# Run the programme 
The programme starts from the [main.py](./main.py) file and can run using Anaconda.
Parameters/number of data files can be changed in main()
How the data files are transformed can be changed in create_clustering_of_directory
Other alterations are made directly in the corresponding functions (e.g. change t-SNE parameter directly in the t-SNE function in [dimensionalityReduction.py](./dimensionalityReduction.py))

## Read in .csv files
Read an entire directory of .csv files using Pandas

## Clean data
 - drop TIME column
 - remove rows with wrong values (missing values)
 - extract columns (1-N)
 - extract rows with minimum of percent non zero values (e.g. 50)
 - compress same attribute columns (1-N)
 - normalize columns (MinMaxScaler)

Other options:
 - remove values with too many empty values

## Dimensionality reduction
(graphs Boolean parameter: True to show graphs, False to hide - default)
 - TSNE - recommended! (extract 2 or 3 components)

 Other options:
 - PCA (extract 2 or 3 components)


## Clustering
(graphs Boolean parameter: True to show graphs, False to hide - default)
 - dbscan_clustering - recommended!
 - optics_clustering - recommended! (with DBSCAN cluster extraction, xi not recommended)

## Cluster Evaluation
 - Silhouette Score (1 -> clustering is better, -1 -> clustering is wrong)
 - Davies Bouldin Score (Lower values (closer to zero) better)
 - Calinksi Harabasz Score (Higher Calinski-Harabasz better)


