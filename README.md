# 🎓 JOSAA College Predictor

A Machine Learning based **JOSAA College Predictor** that estimates college closing ranks and provides personalized college and branch recommendations based on a student's JEE rank, category, quota, gender, and PwD status.

The project uses historical JOSAA counselling data from **2020–2026** and a **Random Forest Regression** model.

## 🚀 Features

* 🎯 JEE Main and JEE Advanced prediction
* 🏫 IIT filtering for JEE Advanced
* 🏛️ NIT, IIIT and GFTI options for JEE Main
* 📊 Historical JOSAA rank analysis
* 🤖 Random Forest based closing-rank prediction
* 🔍 Category, quota, gender and PwD filtering
* 📈 Historical average, trend and volatility features
* 🎯 HIGH / MEDIUM / LOW chance classification
* 🖥️ Interactive Streamlit web application
* 🌐 Deployed using Streamlit
* 📋 College and branch recommendations in table format

---

## 📂 Dataset

**Source:** Kaggle — JOSAA Opening and Closing Ranks Dataset

**Years used:** 2020–2026

The dataset contains information such as:

* Institute
* Academic Program
* Quota
* Seat Type / Category
* Gender
* Opening Rank
* Closing Rank
* Round
* Year

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Extracted yearly JOSAA datasets.
2. Removed data from 2016–2019.
3. Merged yearly and round-wise datasets.
4. Removed invalid rank values.
5. Converted Opening and Closing Rank to numeric values.
6. Removed missing/infinite rank values.
7. Kept counselling rounds up to Round 5.
8. Cleaned column names and categorical values.
9. Created additional features such as `is_pwd` and `Institute_Type`.

Institute types include:

```text
IIT
NIT
IIIT
GFTI
```

---

## ⚙️ Feature Engineering

Historical features were created using previous rank information:

* `hist_avg_closing` — historical average closing rank
* `hist_trend` — historical closing-rank trend
* `hist_volatility` — historical closing-rank variation

Categorical features were encoded using `LabelEncoder`.

### Final Features

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

### Target

```text
Closing_Rank
```

---

## 🤖 Machine Learning

Multiple regression models were evaluated:

| Model             |         MAE |         RMSE |         R² |
| ----------------- | ----------: | -----------: | ---------: |
| **Random Forest** | **1655.24** | **12292.44** | **0.9013** |
| XGBoost           |     1951.64 |     13517.05 |     0.8806 |
| LightGBM          |     1967.25 |     12817.27 |     0.8926 |
| CatBoost          |     2098.24 |     13459.51 |     0.8816 |

### Final Model

**Random Forest Regressor**

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=5,
    max_features=0.8,
    random_state=42,
    n_jobs=2
)
```

### 📊 Model Metrics

The model was evaluated using a **time-based split**, with previous years used for training and **2026 used as the test year**.

```text
Training R² : 0.9868
Testing R²  : 0.9045

Test MAE    : 1655.24
Test RMSE   : 12292.44
```

**R²** measures how much variation in closing rank is explained by the model.

**MAE** represents the average absolute difference between actual and predicted closing ranks.

**RMSE** gives greater weight to larger prediction errors.

---

## 🎯 Prediction System

The application takes:

```text
Exam
Rank
Category
Quota
Gender
PwD Status
```

The prediction workflow is:

```text
User Input
    ↓
Exam & Eligibility Filtering
    ↓
Category / Quota / Gender / PwD Filtering
    ↓
Feature Preparation
    ↓
ML Closing Rank Prediction
    ↓
Compare Student Rank with Predicted Closing Rank
    ↓
HIGH / MEDIUM / LOW
    ↓
College & Branch Recommendations
```

For **JEE Advanced**, IITs are considered.

For **JEE Main**, non-IIT institutes such as NITs, IIITs and GFTIs are considered.

---

## 🖥️ Streamlit Application

The project is deployed as an interactive **Streamlit web application**.

### Landing Page

* Project introduction
* ML/data-driven highlights
* Blue gradient UI
* Feature cards
* **LET'S START** button

### Predictor Dashboard

Users enter their details and receive results in the form:

| College Name | Branch | College Type | Chances |
| ------------ | ------ | ------------ | ------- |

The application uses a clean blue-gradient design with card-based components.

---

## 📁 Project Structure

```text
JOSAA-collage-predictor/
│
├── app.py
├── clg_predictor.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
└── model/
    ├── final_random_forest.pkl
    ├── encoders.pkl
    ├── features.pkl
    └── processed_data.pkl
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* LightGBM
* CatBoost
* Joblib
* Matplotlib
* Seaborn
* Jupyter Notebook
* Streamlit
* Git & GitHub

---

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/ShreshthaPandey/JOSAA-collage-predictor.git
cd JOSAA-collage-predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📌 Limitations

* Predictions are estimates based on historical counselling data.
* Actual JOSAA cutoffs can change based on competition, seats, preferences and counselling trends.
* The model is not an official JOSAA admission prediction system.
* Chance categories are heuristic estimates and not calibrated admission probabilities.
* Prediction quality depends on historical data availability.

---

## 🔮 Future Improvements

* Add college and branch search.
* Add cutoff trend visualizations.
* Add college comparison.
* Improve chance calibration.
* Add newer counselling data.
* Add more personalized recommendations.
* Improve deployment and scalability.

---

## 👨‍💻 Author

**Shreshtha Pandey**

B.Tech CSE — Kamla Nehru Institute of Technology, Sultanpur

### 🔗 GitHub

https://github.com/ShreshthaPandey/JOSAA-collage-predictor

### 🌐 Deployment

The application is deployed using **Streamlit**.

---

## ⭐ Acknowledgement

Dataset sourced from Kaggle's JOSAA Opening and Closing Ranks dataset.

If you find this project useful, consider giving the repository a ⭐.

