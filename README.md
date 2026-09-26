JOSAA College Predictor


## Data Preprocessing

* Downloaded the **JoSAA Opening & Closing Ranks Dataset (2016–2026)** from Kaggle.
* Removed records from **2016–2019** and retained data from **2020–2026**.
* Loaded and inspected individual CSV files using **Pandas**.
* Merged all counselling rounds for each year into a single yearly dataset.
* Added a **`round`** column to identify the counselling round.
* Combined all yearly datasets into a single dataset and added a **`year`** column.
* Checked dataset dimensions, column names, data types, and missing values.
* Identified and handled special rank values containing **`P`**.
* Detected and removed rows containing **infinite (`inf`) rank values**.
* Converted **Opening Rank** and **Closing Rank** into integer format.
* Performed **group-by and sorting operations** for college-wise and rank-based analysis.
* Verified the cleaned dataset before proceeding to further analysis and feature engineering.

