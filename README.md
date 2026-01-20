## Night Phone Usage → Sleep Disorder Predictor

A beginner‑friendly **Machine Learning + Flask** project that predicts sleep disorder levels based on night‑time mobile phone usage and sleep habits.
This project uses **synthetic data**, a **rule‑based labeling system**, and a **Logistic Regression model** wrapped inside a **scikit‑learn pipeline**.

---

## Project Overview

Excessive phone usage at night can seriously affect sleep quality. This project predicts the level of sleep disorder using factors like:

* Phone usage after 10 PM
* Blue light exposure
* Sleep duration
* Night phone pickups
* Attention span and break habits

The prediction is shown through a **Flask web interface** where users can input their habits and get instant results.

---

## Problem Type

**Multi‑Class Classification**

The model predicts one of the following sleep conditions:

| Label | Sleep Condition             |
| ----- | --------------------------- |
| 0     | Healthy Sleep               |
| 1     | Slightly Disturbed Sleep    |
| 2     | Mild Sleep Disorder         |
| 3     | Severe Sleep Disorder       |

---

## Features Used

| Feature Name                           | Type        |
| -------------------------------------- | ----------- |
| phone_usage_after_10pm (minutes)       | Numerical   |
| blue_light_hours                       | Numerical   |
| sleep_duration (hours)                 | Numerical   |
| phone_pickups_night                    | Numerical   |
| night_scrolling (yes/no)               | Categorical |
| wake_up_tired (yes/no)                 | Categorical |
| attention_span (low/medium/high)       | Ordinal     |
| breaks_taken (rare/sometimes/frequent) | Ordinal     |

---

## Dataset

* The dataset is **synthetically generated** using Python
* Labels are assigned using **rule‑based logic** that mimics real‑world sleep behavior
* Dataset size: **500 rows**

File generated automatically:

```
sleep_data.csv
```

---

## Machine Learning Pipeline

The project follows a clean and professional ML workflow:

1. Synthetic data generation
2. Rule‑based target labeling
3. Feature engineering
4. ColumnTransformer

   * StandardScaler for numerical features
   * OrdinalEncoder for ordinal features
   * OneHotEncoder for categorical features
5. Pipeline creation
6. Logistic Regression model training
7. Model serialization using Joblib

Saved model file:

```
sleep_model.pkl
```

---

## Flask Web Application

The Flask app provides a simple web interface where:

* Users enter sleep and phone usage data
* The trained ML model predicts sleep disorder level
* Result is displayed instantly

---

## Model Behavior

* If dataset or model does not exist → app generates & trains automatically
* User input data is **not stored permanently**
* Dataset and trained model **are saved locally**

---

## Why This Project Is Useful

✔ Beginner‑friendly ML project
✔ Real‑life inspired problem
✔ Demonstrates end‑to‑end ML workflow (data → model → deployment) 
✔ Shows ML + Flask integration

---

## Future Improvements

* Store user inputs in database
* Add probability scores
* Improve UI with CSS / Bootstrap
* Use advanced ML models

---

⚠️ Note: The dataset is synthetic and created purely for educational purposes.

Author
Sunaina Imran
Machine Learning & Python Enthusiast




