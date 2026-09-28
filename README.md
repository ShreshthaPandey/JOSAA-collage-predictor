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

# 🎓 JoSAA College & Rank Prediction

## 📌 Project Overview

This project uses **JoSAA Opening and Closing Rank data** to analyze college admission trends and build a machine learning model for rank prediction.

The dataset contains information about institutes, academic programs, quota, category, gender, PwD status, counselling rounds, years, and opening/closing ranks.

---

## ✅ Work Completed

### 1. Data Cleaning

* Inspected categorical columns such as:

  * Institute
  * Academic Program Name
  * Quota
  * Seat Type
  * Gender
* Checked category frequencies using `value_counts()`.
* Removed leading and trailing whitespace from categorical values.

```python
df['Seat Type'] = df['Seat Type'].str.strip()
```

---

### 2. PwD Feature Engineering

Separated PwD information from the `Seat Type` column.

Created a new feature:

```python
df['is_pwd']
```

Encoding:

* `0` → Non-PwD
* `1` → PwD

Example:

| Seat Type  | Category | is_pwd |
| ---------- | -------- | -----: |
| OPEN       | OPEN     |      0 |
| OPEN (PwD) | OPEN     |      1 |
| SC         | SC       |      0 |
| SC (PwD)   | SC       |      1 |

---

### 3. Category Extraction

Removed `(PwD)` from `Seat Type` to create a clean category column.

```python
df['category'] = (
    df['Seat Type']
    .str.replace(r'\s*\(PwD\)', '', regex=True)
    .str.strip()
)
```

Main categories:

```text
OPEN
OBC-NCL
SC
EWS
ST
```

PwD information is stored separately in `is_pwd`.

---

### 4. Categorical Data Analysis

Analyzed the distribution of:

* Seat Type
* Category
* Quota
* Academic Program Name
* Gender
* PwD status

Used frequency counts and bar charts to understand the dataset.

For columns with many unique values, such as Academic Program Name, a **Top-N visualization** was used for better readability.

---

### 5. Year & Round

`Year` and `Round` were identified as naturally ordered numerical features.

They can be kept as numerical values rather than unnecessarily applying `LabelEncoder`.

```python
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Round'] = pd.to_numeric(df['Round'], errors='coerce')
```

---

### 6. Feature Engineering

Created/considered useful rank-based features:

```python
df['rank_range'] = df['Closing Rank'] - df['Opening Rank']

df['avg_rank'] = (
    df['Opening Rank'] + df['Closing Rank']
) / 2

df['rank_ratio'] = (
    df['Closing Rank'] / df['Opening Rank']
)
```

Also considered log transformations:

```python
df['log_opening_rank'] = np.log1p(df['Opening Rank'])
df['log_closing_rank'] = np.log1p(df['Closing Rank'])
```

> ⚠️ Features directly derived from the target variable must not be used for prediction because they can cause **data leakage**.

---

## 📊 Current Important Columns

```text
Institute
Academic Program Name
Quota
Seat Type
Gender
Opening Rank
Closing Rank
Round
Year
is_pwd
category
```



# 🎓 JOSAA College Predictor

A Machine Learning-based college predictor using **JOSAA Opening & Closing Rank data (2020–2026)** to predict closing ranks and help estimate suitable college/branch options.

## 📊 Dataset

* **Source:** JOSAA Opening & Closing Ranks Dataset
* **Years Used:** 2020–2026
* **Original Records:** 366,914
* **Data after cleaning/filtering:** ~259K records
* **Rounds Used:** 1–5
* **Institutes:** 135
* **Academic Programs:** 292
* **Quotas:** 4
* **Gender Categories:** 2
* **Categories:** 5
* **Institute Types:** 4

## 🧹 Data Preprocessing

* Removed data from **2016–2019**
* Removed invalid ranks containing `P`
* Converted Opening/Closing Rank to numeric
* Removed `NaN` and infinite rank values
* Restricted counselling rounds to **1–5**
* Cleaned whitespace from categorical columns
* Created `is_pwd` feature from quota information

## ⚙️ Feature Engineering

Created historical features using previous rank data:

* `hist_avg_closing` — rolling historical average closing rank
* `hist_trend` — historical closing-rank trend
* `hist_volatility` — historical closing-rank variation

### Final Model Features

```text
institute_enc
branch_enc
quota_enc
gender_enc
category_enc
inst_type_enc
Opening_Rank
Round
Year
is_pwd
hist_avg_closing
hist_trend
hist_volatility
```

**Target:** `Closing_Rank`

## 🤖 Models Compared

* Random Forest
* XGBoost
* LightGBM
* CatBoost

### Earlier Model Comparison

| Model         |         MAE |         RMSE |         R² |
| ------------- | ----------: | -----------: | ---------: |
| Random Forest | **1665.11** | **12073.89** | **0.8984** |
| XGBoost       |     1807.16 |     12825.03 |     0.8853 |
| LightGBM      |     1896.67 |     12635.64 |     0.8887 |
| CatBoost      |     1886.52 |     12872.45 |     0.8845 |

Random Forest was the best model in this evaluation based on MAE, RMSE and R².

## 💾 Model Saving

Saved the trained model and preprocessing information using Joblib:

```text
model/
├── random_forest_model.pkl
├── encoders.pkl
└── features.pkl
```

* `random_forest_model.pkl` → trained ML model
* `encoders.pkl` → categorical encoders
* `features.pkl` → feature order used during training

## 🚀 Prediction Flow

```text
User Input
   ↓
Category + Rank + Quota + Gender + PwD
   ↓
Feature Encoding
   ↓
Historical Features
   ↓
Random Forest Model
   ↓
Predicted Closing Rank
   ↓
College + Branch + Round + Chance
```

## 🛠️ Tech Stack

**Python | Pandas | NumPy | Scikit-learn | Random Forest | XGBoost | LightGBM | CatBoost | Joblib | Streamlit**

## 📌 Repository

GitHub: `ShreshthaPandey/JOSAA-collage-predictor`

