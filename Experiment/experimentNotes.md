# Problem: 
Keine eindeutige Ergebnisse

# Read in files
pandas - concat

# Clean data
 - drop TIME column
 - (remove values with too many empty values)
 - remove rows with wrong values (missing values)
 - compress same attribute columns (1-N)
 - normalize columns (MinMaxScaler sklearn)

# Dimensionality reduction
 - PCA (extract 2 or 3 components)
 - TSNE (extract 2 or 3 components)
    * always different results (not enough data?)

# Try without dimensionality reduction
 - todo

# Clustering
 - spectral_clustering (3 clusters)
    * not so good - not sure how many clusters
 - dbscan_clustering
    * pca - only one cluster
    * tsne - a few clusters - but not sure if natural 
 - optics_clustering
 - aggolomerative_clustering

# Cluster Evaluation
 - Silhouette Score (-1, 1 -> 1 means clustering is correct, -1 means it's wrong)
 - Davies Bouldin Score (Zero is the lowest possible score. Values closer to zero indicate a better partition.)
 - Calinksi Harabasz Score (higher Calinski-Harabasz score relates to a model with better defined clusters)




# FRAGEN
* TIME?
* 1-N zusammenfassen mit Median?
* TSNE - immer andere Ergebnisse
* Scores stimmen nicht überein (Silhouette score, Davies Bouldin, Calinski Harabasz)
* wie am besten dbscan (Density based methods) evaluieren? vl eher test personen fragen?

* Was genau im Diskussion teil (warum ich glaube, dass eine gewisse zeit besser/schlechter ist?)
* Details für die aufnahme von den daten - wie viele personen, welches alter, womit aufgenommen…? notwendig? !!
* muss ich erklären was accel.. ist?
* min max scaler - also on fields where already between 0.0 and 1.0?
* wurden im email alle aufgelisteten features verwendet (Gyro, Telefon, Zeitpunkt Nachrichten, Datenvolumen pro Zeitinhalt, An- Ausschaltzeiten)?
* Experiment - wie erwähnt in section... zusammenfassung zitieren?
* Wie viel detail für die verwendeten algorithmen?
* Theory vs Literature Review, Experiment vs Method
* neues Termin ausmachen


relative Bewegung des Handys (Gyro und Accel)
Zeitpunkt und Dauer von Telefongesprächen ohne Speicherung der Nummern
Zeitpunkt von Nachrichten (z.B.: SMS, WhatsApp) ohne Erhebung von identifizierender Information wie Inhalt, Adressen, Nummern)
Bildschirmaktivität (sog. Touch-Events)
Screen-On-Time (Display beleuchtet); Umgebungshelligkeit
Datenvolumen pro Zeiteinheit (summarischer Wert aller Smartphone Aktivitäten im Internet)
An- und Ausschaltzeiten des Handys.