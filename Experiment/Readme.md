Folder Structure:
	"aggregated": contains data aggregated for each user (= filename)
		- "1h": Data aggregated in 2.5h intervals. Each row is an aggregation of 1h in the past in 4 15min lags.
		- "3h": Data aggregated in 1.5h intervals. Each row is an aggregation of 3h in the past in 6 30min lags.

	"clusters": contains the cluster index for each data row of the aggregated data. Clusters were automatically detected by reducing the dimensionality of the aggregated data rows to 2 dimensions using t-SNE and then applying simple spectral clustering with k = 3.

Columns:
	TIME: 			The timestamp of where the data is aggregated from (in format YYYY-DD-MM hh:mm:ss)
	ACC (1-N): 		Accelerometer values (we calculate the "average jerk"; N averages of lag-interval minutes over the aggregation interval in the past)
	-> Missing Values: ? average/median

	AUDIO (1-N): 		Audio volume (same lagged averages as Accelerometer)
	-> Missing Values: 0 or average

	SCRN (1-N): 		Percentage of Screen being on in lag-interval minutes (same lagged intervals as Accelerometer)
	-> Missing Values: 0 or average

	NOTIF (1-N): 		Notification amount in lag-interval minutes (same lagged intervals as Accelerometer)
	-> Missing Values: 0 or average (if not activated)

	LIGHT (1-N): 		Light Sensor values (same lagged intervals as Accelerometer)
	-> Missing Values:  0

	APP_COM (1-N):		App usage of category 'Communication' in percent of lag-interval minutes (i.e. if apps of this category were used for 5 minutes in the past 15 minutes the value is 0.66666)
	-> Missing Values: 0

	APP_VID (1-N):		App usage of category 'Video_Players', same principle as APP_COM
	-> Missing Values: 0

	APP_OTHER (1-N):	App usage of all other categories (excluding 'Video_Players' and 'Communication')
	-> Missing Values: ?