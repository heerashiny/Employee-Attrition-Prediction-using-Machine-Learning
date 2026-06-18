# Employee Attrition Prediction using Machine Learning

## Problem Statement

Employee attrition is a major challenge for organizations, leading to increased recruitment costs, productivity loss, and disruption of business operations.

The objective of this project is to predict whether an employee is likely to leave the company based on HR-related factors and identify the key drivers of attrition. These insights can help organizations take proactive measures to improve employee retention.

---

## Dataset

**IBM HR Analytics Employee Attrition Dataset**

* Source: Kaggle
* Records: 1,470 employees
* Features: 35 employee-related attributes
* Target Variable: Attrition (Yes/No)

The dataset contains information related to employee demographics, compensation, job satisfaction, work-life balance, performance, and employment history.

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

* Examined dataset structure and feature distributions
* Checked for missing values and data quality issues
* Analyzed attrition class distribution
* Identified important trends and relationships

### 2. Data Preprocessing

* Removed unnecessary columns
* Encoded categorical variables
* Prepared feature matrix and target variable
* Performed train-test split (80:20)

### 3. Handling Class Imbalance

The target variable was imbalanced:

* No Attrition: ~83%
* Attrition: ~17%

To improve detection of employees likely to leave, a Decision Tree model was trained using:

```python
class_weight='balanced'
```

### 4. Model Training

The following machine learning models were trained and evaluated:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Naive Bayes
* Decision Tree Classifier

### 5. Model Evaluation

Models were evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report
* Recall Score
* F1 Score

---

## Model Comparison

| Model                    | Accuracy | Recall (Attrition) | F1 Score (Attrition) |
| ------------------------ | -------- | ------------------ | -------------------- |
| Logistic Regression      | 0.88     | 0.46               | 0.51                 |
| KNN                      | 0.88     | 0.13               | 0.22                 |
| SVM                      | 0.90     | 0.26               | 0.40                 |
| Naive Bayes              | 0.69     | 0.67               | 0.36                 |
| Decision Tree (Balanced) | 0.79     | 0.38               | 0.32                 |

### Observations

* SVM achieved the highest accuracy (90%).
* Naive Bayes achieved the highest recall for attrition detection.
* Logistic Regression achieved the best F1-score, providing the best balance between precision and recall.
* Decision Tree provided the highest interpretability through visual decision rules.

---

## Feature Importance Analysis

Feature importance analysis was performed using the Decision Tree model.

### Top Factors Influencing Attrition

1. MonthlyIncome
2. OverTime
3. HourlyRate
4. TotalWorkingYears
5. DailyRate
6. DistanceFromHome
7. Age
8. MonthlyRate
9. PercentSalaryHike
10. YearsWithCurrManager

While MonthlyIncome ranked highest in feature importance score, OverTime was selected as the root split by the Decision Tree, indicating it provides the cleanest separation between classes.

### Key Insight

Compensation-related factors, overtime workload, and employee experience were identified as the strongest drivers of employee attrition.

---

## Decision Tree Visualization

The trained Decision Tree was visualized using:

```python
plot_tree()
```

The root node of the tree was:

```text
OverTime_Yes
```

This indicates that overtime was the most influential feature used by the model when predicting employee attrition.

---

## Key Findings

* Overtime is one of the strongest indicators of employee attrition.
* Compensation-related variables significantly impact employee retention.
* Employee experience and tenure influence attrition behavior.
* Handling class imbalance improved the model's ability to identify employees likely to leave.
* Logistic Regression provided the strongest overall performance based on F1-score.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

---

## Future Improvements

* Hyperparameter tuning using GridSearchCV
* Random Forest and XGBoost implementation
* SHAP-based model explainability
* Deployment using Streamlit or Flask
* Real-time employee attrition prediction dashboard

---

## Author
[LinkedIn Profile](https://www.linkedin.com/in/heera-shiny-234838297?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
