# 04 - Classification Modeling

This notebook compares classification models for predicting a one-year adverse waitlist outcome.

The final model was XGBoost without CPRA.

On the 2024 test set:

- ROC-AUC: 0.6742
- PR-AUC: 0.0860
- Recall: 48.22% at the threshold selected using validation data
- Top-decile lift: 2.53x

I also checked subgroup performance and model explanations. The output folder includes the model comparison, fairness tables, feature importance results, SHAP summaries, and validation-versus-test results.

The results show useful risk separation, but the model is not accurate enough to be treated as a clinical decision system.
