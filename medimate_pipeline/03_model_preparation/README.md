# 03 - Model Preparation

This notebook turns the cleaned dataset into the datasets used for classification modeling.

The classification target is a one-year adverse waitlist outcome: death or removal because the patient became too sick for transplant.

The data was split by time:

- Training: 2015-2022
- Validation: 2023
- Test: 2024

Patient IDs were also kept separate across the splits so the same patient could not appear in both training and evaluation data.

I prepared versions with and without CPRA. The final modeling pipeline does not use CPRA because recent years have very high missingness and it provided very little additional validation performance.

Only the public feature lists are kept in the outputs folder. The matrices, patient metadata, labels, and fitted preprocessors are not published.
