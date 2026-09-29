

# 🎓 JOSAA College Predictor

A Machine Learning based **JOSAA College Predictor** that estimates college closing ranks and provides personalized college and branch recommendations based on a student's JEE rank, category, quota, gender, and PwD status.

The project uses historical JOSAA counselling data from **2020–2026** and a **Random Forest Regression** model.

---

## 🚀 Features

* 🎯 JEE Main and JEE Advanced prediction
* 🏫 IIT filtering for JEE Advanced
* 🏛️ NIT, IIIT and GFTI options for JEE Main
* 📊 Historical JOSAA rank analysis
* 🤖 Random Forest based closing-rank prediction
* 🔍 Category, quota, gender and PwD filtering
* 📈 Historical average, trend and volatility features
* 🎯 HIGH / MEDIUM / LOW chance classification
* 🖥️ Interactive Streamlit web interface
* 📋 College and branch recommendations in table format

---

## 📂 Dataset

**Source:** Kaggle — JOSAA Opening and Closing Ranks Dataset

**Years used:** 2020–2026

Original data contained counselling information such as:

* Institute
* Academic Program
* Quota
* Seat Type / Category
* Gender
* Opening Rank
* Closing Rank
* Round
* Year

Rows containing invalid rank values such as `P` and missing/infinite ranks were cleaned before model training.

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
9. Created additional features such as:

   * `is_pwd`
   * `Institute_Type`

Institute types include:

* IIT
* NIT
* IIIT
* GFTI

---

## ⚙️ Feature Engineering

Historical features were created without using the current/future closing rank:

* `hist_avg_closing` — rolling historical average closing rank
* `hist_trend` — historical closing-rank trend
* `hist_volatility` — historical closing-rank variation

Categorical features were encoded using `LabelEncoder`.

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

### Target

```text
Closing_Rank
```

---

## 🤖 Machine Learning Models

Several regression models were evaluated:

| Model         |         MAE |     RMSE |         R² |
| ------------- | ----------: | -------: | ---------: |
| Random Forest | **1655.24** | 12292.44 | **0.9013** |
| XGBoost       |     1951.64 | 13517.05 |     0.8806 |
| LightGBM      |     1967.25 | 12817.27 |     0.8926 |
| CatBoost      |     2098.24 | 13459.51 |     0.8816 |

Based on the time-based evaluation, **Random Forest Regression** was selected as the final model.

Final configuration:

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

### Final Evaluation

```text
Training R² : 0.9868
Testing R²  : 0.9045
```

The model was evaluated using a time-based split, with historical years used for training and **2026 as the test year**.

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

The system then:

```text
User Input
    ↓
Filter eligible institutes
    ↓
Apply category/quota/gender/PwD filters
    ↓
Generate model predictions
    ↓
Compare student rank with predicted closing rank
    ↓
Assign Chance
    ↓
Display College + Branch
```

Chance levels:

* 🟢 **HIGH**
* 🟡 **MEDIUM**
* 🔴 **LOW**

These are guidance categories based on the predicted closing rank and are **not guaranteed admission probabilities**.

---

## 🖥️ Streamlit Application

The application contains two main screens.

### 1. Landing Page

* Project introduction
* ML/data-driven highlights
* Blue gradient UI
* **LET'S START** button

### 2. Predictor Dashboard

Users enter their JEE details and receive:

| College Name | Branch | College Type | Chances |
| ------------ | ------ | ------------ | ------- |

The interface uses a clean blue-gradient theme with card-based components.

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

Model files are kept outside GitHub using `.gitignore`.

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

### 1. Clone the repository

```bash
git clone https://github.com/ShreshthaPandey/JOSAA-collage-predictor.git
cd JOSAA-collage-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Make sure the Scikit-Learn version matches the model's training environment.

```text
scikit-learn==1.9.0
```

### 3. Run Streamlit

```bash
streamlit run app.py
```

---

## 📌 Limitations

* Predictions are estimates based on historical counselling data.
* Actual JOSAA cutoffs can change due to competition, seats, preferences, and counselling trends.
* The model should not be treated as an official JOSAA admission prediction.
* Chance categories are heuristic rather than calibrated admission probabilities.
* Historical data availability affects prediction quality.

---

## 🔮 Future Improvements

* Add college/branch search and advanced filters.
* Add cutoff trend visualizations.
* Add personalized college comparison.
* Add probability calibration for admission chances.
* Add more recent counselling data.
* Deploy the application online.
* Add separate prediction models for different institute types.

---

## 👨‍💻 Author

**Shreshtha Pandey**

B.Tech CSE — Kamla Nehru Institute of Technology, Sultanpur

🔗 **GitHub:**
https://github.com/ShreshthaPandey/JOSAA-collage-predictor

---

## ⭐ Acknowledgement

Dataset sourced from Kaggle's JOSAA Opening and Closing Ranks dataset.

If you find this project useful, consider giving the repository a ⭐.
