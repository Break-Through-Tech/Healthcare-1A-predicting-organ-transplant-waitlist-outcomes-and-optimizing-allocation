
# 🩺 MediMate | Kidney Transplant Waitlist Risk Analysis

### Break Through Tech AI Studio — Fall 2026
**Industry Partner:** MediMate Foundation

**Project:** Predicting Organ Transplant Waitlist Outcomes and Optimizing Allocation

[![GitHub](https://img.shields.io/badge/GitHub-Project-181717?logo=github)](https://github.com/Break-Through-Tech/Healthcare-1A-predicting-organ-transplant-waitlist-outcomes-and-optimizing-allocation)
[![Live Dashboard](https://img.shields.io/badge/Live-Dashboard-blue)](https://healthcare-1a-predicting-organ.onrender.com/)

---

## 📌 Project Overview

This project was developed through the Break Through Tech AI Studio in collaboration with MediMate Foundation.

We investigated kidney transplant waitlist outcomes using machine learning and survival analysis to better understand adverse waitlist risks.

The project focuses on:
- Identifying candidates at increased risk of adverse waitlist outcomes.
- Analyzing how adverse risk changes over time.
- Understanding factors associated with model predictions.
- Evaluating differences in model performance across patient subgroups.
- Communicating findings through an interactive dashboard.

The models are intended for research and educational analysis, not clinical decision-making or organ allocation.

## 🎯 Objectives

1. Explore kidney waitlist data and identify important patient characteristics.
2. Prepare a modeling-ready dataset while minimizing data leakage.
3. Develop and evaluate adverse-outcome classification models.
4. Apply survival analysis to time-to-event outcomes.
5. Examine fairness and model explainability.
6. Present the findings through an accessible web dashboard.

## 📊 Dataset

**Source:** Organ Procurement and Transplantation Network (OPTN) STAR data.

**Working cohort:** Kidney-alone waitlist candidates registered in 2015 or later.

**Primary analytic extract:** 494,862 rows, as documented in `data/README.md`.

The analysis considers characteristics including:
- Age
- Blood type
- Dialysis status
- Functional status
- Region
- Other available candidate and waitlist characteristics

### Data Privacy

The raw OPTN dataset and patient-level extracts are governed by a Data Use Agreement.

They are not provided for public download through this repository.

Only permitted code, documentation, and aggregate research outputs should be published.

See [Data Documentation](data/README.md) for the access requirements.

## 🔬 Methodology

The analysis was organized into six stages.

### 1. Exploratory Data Analysis

Explored patient demographics, waitlist outcomes, missing values, repeated registrations, follow-up time, and potential predictive features.

### 2. Data Cleaning

Prepared modeling data by cleaning patient characteristics, dates, survival times, and other variables.

Checked invalid timelines and potential target leakage.

### 3. Model Preparation

Created a one-year adverse waitlist outcome target.

Used a temporal evaluation strategy:

| Partition | Period |
|---|---|
| Training | 2015–2022 |
| Validation | 2023 |
| Testing | 2024 |

CPRA was excluded from the final models due to substantial missingness in recent years and limited incremental improvement.

### 4. Classification Modeling

Compared classification approaches and selected XGBoost as the final classification model.

Evaluated discrimination, recall, risk concentration, and subgroup performance.

### 5. Survival Analysis

Used survival modeling to examine adverse outcomes over time.

The analysis accounted for transplantation as a competing event and evaluated the final XGBoost survival model using concordance.

### 6. Final Analysis and Visualization

Combined classification and survival findings, examined model explanations, and presented aggregated results through the MediMate dashboard.

## 📈 Results and Evaluation

### Classification Performance

Reported results on the 2024 test set:

| Metric | Result |
|---|---|
| ROC-AUC | 0.6742 |
| PR-AUC | 0.0860 |
| Recall | 48.22% |
| Top-decile lift | 2.53× |

### Survival Model Performance

| Metric | Result |
|---|---|
| C-index | 0.6784 |

### One-Year Adverse Incidence by Risk Group

| Risk Group | Adverse Incidence |
|---|---|
| Low | 1.70% |
| Medium | 3.71% |
| High | 6.60% |

The high-risk group had approximately 3.87 times the observed one-year adverse incidence of the low-risk group.

### Key Findings

Age and functional status were among the strongest shared factors across the models.

Region, ethnicity code, and dialysis status also contributed to the analysis.

The models demonstrate risk stratification potential, while their discrimination and calibration limitations require further investigation.

## 📉 Visualizations and Dashboard

The project dashboard presents:

- Exploratory data analysis
- Data preparation and modeling methodology
- Classification performance
- Survival and competing-risk analysis
- Fairness and subgroup evaluation
- Model explainability
- Final research findings

**[🚀 Explore the Live Dashboard](https://healthcare-1a-predicting-organ.onrender.com/)**

## 🗂️ Repository Structure

```text
.
├── EDA/
├── data/
│   └── README.md
├── medimate_dashboard/
│   └── README.md
├── medimate_pipeline/
│   └── README.md
├── notebooks/
├── README.md
├── requirements.txt
└── Challenge-Project-Overview.md
```

- `EDA/`: Exploratory analysis materials.
- `data/`: Data documentation and access information.
- `medimate_pipeline/`: Modeling workflow and project results.
- `medimate_dashboard/`: Interactive visualization application.
- `notebooks/`: Jupyter notebooks.
- `requirements.txt`: Python dependencies.

## ⚙️ Installation and Setup

Clone the repository:

```bash
git clone https://github.com/Break-Through-Tech/Healthcare-1A-predicting-organ-transplant-waitlist-outcomes-and-optimizing-allocation.git

cd Healthcare-1A-predicting-organ-transplant-waitlist-outcomes-and-optimizing-allocation
```

Create a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies after the requirements file has been populated and verified:

```bash
python -m pip install -r requirements.txt
```

Refer to the notebooks and pipeline documentation for the analysis stages.

### Reproducing the Results

The notebooks and project pipeline document the analytical workflow.

Full model reproduction requires authorized access to the OPTN data under the applicable Data Use Agreement.

Raw patient records, patient-level predictions, and trained model artifacts are not included in the public repository.

See:
- [Pipeline Documentation](medimate_pipeline/README.md)
- [Dashboard Documentation](medimate_dashboard/README.md)
- [Data Access Documentation](data/README.md)

## 👥 Team and Contributions

This project was developed collaboratively as part of Break Through Tech AI Studio.

| Member | GitHub |
|---|---|
| Prince Subedi | [@Dev-PrinceSubedi](https://github.com/Dev-PrinceSubedi) |
| Arjun Pun Magar | [@arjunmagarcode](https://github.com/arjunmagarcode) |
| Julia Zhao | [@JuliaaaZ123](https://github.com/JuliaaaZ123) |
| Sophie Zhuang | [@sophie-26](https://github.com/sophie-26) |
| Lincoln Chitswa | [@Lincolntinodaishe](https://github.com/Lincolntinodaishe) |
| Christian Germain | [@loldjy123](https://github.com/loldjy123) |

### Individual Contributions

**Prince Subedi** — Project Lead | Led and collaborated with team members across data analysis, preprocessing, feature engineering, machine learning, predictive modeling, model evaluation, survival analysis, fairness assessment, explainability, dashboard development, deployment, and documentation, integrating individual contributions into a unified end-to-end healthcare AI solution.

**Arjun Pun Magar** — EDA, Evaluation Framework, and Model Validation | Conducted column-level EDA across 494,862 OPTN kidney waitlist records and 40 variables, establishing the shared data foundation the full team built models on. Designed and implemented the standardized evaluation framework before model training, computing ROC-AUC, PR-AUC, Brier Score, Calibration Curve, Precision, Recall, and F1-Score to ensure fair comparison across all classifiers on a 93%+ class-imbalanced dataset. Led model evaluation review and pipeline integration during the November milestone, verifying that classification results (ROC-AUC 0.674, top-decile lift 2.53x), survival outputs (C-index 0.678), and SHAP fairness audits were reproducible and correctly documented across all team notebooks before final submission.

**Julia Zhao** — [Contribution]

**Kevin Yang** — [Contribution]

**Sophie Zhuang** — [Contribution]

**Lincoln Chitswa** — [Contribution]

**Christian Germain** — [Contribution]


Other team members' individual contributions should also be documented based on their actual work.

## 🚀 Limitations and Future Improvements

- Improve classification discrimination and probability calibration.
- Investigate sources of false positives and false negatives.
- Evaluate model performance over additional time periods.
- Expand subgroup fairness analysis.
- Further investigate competing-risk modeling assumptions.
- Improve reproducibility using permitted data and documented environments.
- Assess generalizability to different patient populations and allocation policies.

## 🙏 Acknowledgements

Thanks to MediMate Foundation, Break Through Tech AI Studio, Challenge Advisor Deepti Bahel, AI Studio Coach Nagalakshmi Pulivarthi, and all project collaborators.

## 📄 References

- [OPTN Data](https://optn.transplant.hrsa.gov/data/)
- [Scientific Registry of Transplant Recipients](https://www.srtr.org/)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)

---

**Disclaimer:** This project is for educational and research purposes only. It is not designed or validated to make individual medical decisions or determine transplant allocation.
