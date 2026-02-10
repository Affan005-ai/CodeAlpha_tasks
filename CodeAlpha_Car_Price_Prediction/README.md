# 🚗 Car Price Prediction using Machine Learning

## 📌 Project Overview

This project predicts the **selling price of cars** based on real-world features such as year of manufacture, fuel type, transmission, ownership, and usage.  
It demonstrates an **end-to-end machine learning workflow** including data preprocessing, feature engineering, model training, evaluation, and model selection.

This project was completed as part of the **CodeAlpha Data Science Internship**.

---

## 🎯 Objectives

- Predict car selling prices using regression models
- Handle real-world structured data
- Apply preprocessing and feature engineering
- Compare multiple ML models and select the best
- Evaluate performance using standard regression metrics

---

## 📂 Dataset

The dataset contains ~300 records with the following features:

| Feature       | Description               |
| ------------- | ------------------------- |
| Car_Name      | Name of the car           |
| Year          | Manufacturing year        |
| Present_Price | Current showroom price    |
| Driven_kms    | Total distance driven     |
| Fuel_Type     | Petrol / Diesel           |
| Selling_type  | Dealer / Individual       |
| Transmission  | Manual / Automatic        |
| Owner         | Number of previous owners |
| Selling_Price | Target variable           |

---

## 📊 Exploratory Data Analysis (EDA)

EDA is performed in `notebooks/EDA.ipynb` and includes:

- Price distribution analysis
- Correlation heatmaps
- Impact of fuel type & transmission on price
- Depreciation trends by year

📈 **Charts Used**

- Histograms
- Boxplots
- Heatmaps
- Bar plots

---

## 🧠 ML Workflow

- Data → Preprocessing → Feature Engineering →
- Model Training → Evaluation → Best Model Selection

---

## ⚙️ Models Used

- Random Forest
- Decision Tree
- Gradient Boosting
- Linear Regression
- K-Neighbours Regressor
- XGB Regressor
- Ada Boost Regressor

The **best model is selected automatically** based on R² score.

---

## 📐 Evaluation Metrics

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

---

## 📁 Project Structure

```text
Car-Price-Prediction/
│
├── data/
│   └── car_data.csv        # Raw dataset
│
├── notebooks/
│   └── EDA.ipynb           # Exploratory analysis & charts
│
├── src/
│   ├── preprocessing.py   # Data cleaning & feature engineering
│   ├── model.py           # Model training & selection
│   ├── evaluation.py      # Model evaluation metrics
│   └── __init__.py        # Package initializer
│
├── models/
│   ├── best_model.pkl     # Trained ML model
│   └── preprocessor.pkl  # Saved preprocessing pipeline
│
├── main.py                # Runs full ML pipeline
└── README.md


---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py
```
