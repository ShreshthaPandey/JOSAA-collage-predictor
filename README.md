JOSAA College Predictor


###Data Preprocessing
Downloaded the JoSAA Opening & Closing Ranks Dataset (2016–2026) from Kaggle.
Removed data files from 2016–2019 and retained 2020–2026 data.
Loaded individual CSV files using Pandas.
Combined all rounds of each year into a single yearly dataset.
Added a round column to identify the counselling round.
Combined all yearly datasets into a single dataset.
Added a year column to preserve the counselling year.
Checked dataset dimensions, column names, and missing values.
Identified special rank values such as values containing P and removed those records where required.
Detected infinite (inf) values in Opening/Closing Rank columns and removed affected records.
Converted Opening Rank and Closing Rank into integer data types.
Sorted and analyzed rank data using Pandas groupby() and sort_values().
Performed college-wise frequency/count analysis to understand the distribution of counselling records.
Final Dataset
Years: 2020–2026
Rounds: Combined
Additional Features: year, round
Rank Features: Opening Rank, Closing Rank
