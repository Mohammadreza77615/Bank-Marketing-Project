# Bank-Marketing-Project
This project conducts an exploratory data analysis (EDA) of bank marketing data to identify the key factors influencing customers’ subscription to term deposits. The process includes data cleaning, feature engineering, and visualization. Finally, a Random Forest model is trained and evaluated to predict the likelihood of customer subscription.



# 📊 Bank Marketing EDA & Subscription Prediction

This project conducts an Exploratory Data Analysis (EDA) on the Portuguese bank marketing dataset. The goal is to uncover factors that influence a customer's decision to subscribe to a term deposit. Following the analysis, a machine learning model (Random Forest) is developed to predict subscription outcomes.

## 📝 Project Overview

The core of this project is the `Bank Marketing EDA.ipynb` notebook, which systematically investigates the dataset to answer key questions, such as:

* Which demographic groups (age, job) are most likely to subscribe?
* How do economic indicators (e.g., euribor rate, consumer confidence) impact campaign success?
* What is the effect of previous campaign contacts (`poutcome`) on the current campaign's result?

## 🗂️ The Dataset

The dataset used (`bank-additional-full.csv`) contains over 41,000 records and 21 features, detailing:

* **Customer Information:** Age, job, marital status, education, credit default, housing loan, personal loan.
* **Campaign Contact:** Contact type (cellular, telephone), month, day of the week, contact duration.
* **Other Campaign Data:** Number of contacts (`campaign`), days since last contact (`pdays`), outcome of previous campaign (`poutcome`).
* **Economic & Social Indicators:** Employment variation rate, consumer price index, consumer confidence index, Euribor 3-month rate, number of employees.
* **Target Variable:** `y` - Has the client subscribed to a term deposit? (yes/no).

## 🚀 Analysis & Modeling Workflow

The notebook follows a structured approach to data analysis and machine learning:

1.  **Load & Inspect:**
    * Import essential libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `sklearn`).
    * Load the data and perform an initial review (`df.info()`, `df.describe()`).

2.  **Data Cleaning:**
    * Identify and remove duplicate entries.
    * Manage 'unknown' values in categorical columns (e.g., imputing with the mode).
    * Detect and handle outliers in numerical features (e.g., `age`, `campaign`, `duration`) using Boxplots and the IQR method.

3.  **Exploratory Data Analysis (EDA):**
    * Analyze the overall campaign success rate (approx. 11.3%).
    * Visualize the distribution and success rate based on various features, including:
        * Demographics (Age bins, job, education, marital status)
        * Financial Status (Housing/personal loans)
        * Campaign Context (Contact type, month, day of week)
        * Economic Indicators (`emp.var.rate`, `cons.conf.idx`)

4.  **Feature Engineering:**
    * Create new feature groups (e.g., `income_group`, `edu_group`).
    * Apply Cyclical Encoding to temporal features (`month`, `day_of_week`).
    * Encode the binary target variable `y` (yes/no) to (1/0).

5.  **Model Preparation & Training:**
    * Apply One-Hot Encoding to remaining categorical features.
    * Split the data into training and testing sets (`train_test_split`).
    * Scale numerical features using `StandardScaler`.
    * Train a `RandomForestClassifier` model.

6.  **Model Evaluation:**
    * Perform hyperparameter tuning using `RandomizedSearchCV` to optimize for the F1-score.
    * Assess model performance by comparing F1-score and Accuracy on both training and test sets to check for overfitting.
    * Generate and visualize a **Confusion Matrix** to evaluate the final model's classification accuracy, precision, and recall.

## 🛠️ Core Libraries Used

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `plotly`
* `scipy`
* `scikit-learn (sklearn)`

## 🏃 How to Run

1.  Ensure you have the `bank-additional-full.csv` dataset in the same directory as the notebook.
2.  Open `Bank Marketing EDA.ipynb` in a Jupyter environment (Jupyter Lab, VS Code, etc.).
3.  Run the cells sequentially to reproduce the analysis and model training.