# MediMate Kidney Waitlist Analysis
## Live Website

[MediMate Kidney Waitlist Risk Explorer](https://healthcare-1a-predicting-organ.onrender.com/)

Interactive website for exploring the project results, including classification, survival analysis, fairness, explainability, and the final findings.
EOF

This project studies kidney transplant waitlist outcomes using OPTN kidney-alone waitlist data.

The main goal was to identify patients who may be at higher risk of dying or being removed from the waitlist because they became too sick for transplant, while also studying how that risk changes over time.

## What I Did

The project was completed in six main steps:

### 1. Exploratory Data Analysis
I first explored the waitlist data to understand the patient population, outcomes, missing values, repeated registrations, follow-up time, and differences across factors such as age, dialysis status, blood type, functional status, and region.

### 2. Data Cleaning
I created a modeling-ready dataset by cleaning dates, BMI, functional status, CPRA, survival times, and other variables. I also checked for invalid timelines and possible target leakage.

### 3. Model Preparation
I created a one-year adverse outcome target and used patient-safe temporal splits:

- Training: 2015-2022
- Validation: 2023
- Test: 2024

CPRA was excluded from the final models because recent years had very high missingness and it provided very little improvement.

### 4. Classification Modeling
I compared several classification models and selected XGBoost as the final model.

2024 test results:

- ROC-AUC: 0.6742
- PR-AUC: 0.0860
- Recall: 48.22%
- Top-decile lift: 2.53x

I also evaluated subgroup performance and model explanations.

### 5. Survival Modeling
I used survival analysis to study how adverse risk changes over time.

Transplant was treated as a competing event.

The final XGBoost survival model reached a 2024 test C-index of 0.6784.

One-year adverse incidence:

- Low risk: 1.70%
- Medium risk: 3.71%
- High risk: 6.60%

The high-risk group had about 3.87 times the one-year adverse incidence of the low-risk group.

### 6. Final Analysis
I combined the classification and survival results and created the final tables and figures.

Age and functional status were the strongest shared factors across both models. Region, ethnicity code, and dialysis status were also important.

## Repository Structure

Each stage contains the notebook and the outputs produced by that notebook.

Raw OPTN data, patient-level predictions, preprocessing matrices, and trained model files are not included in this repository.

This project is for research and analysis only. The models are not intended to make clinical or organ allocation decisions.
