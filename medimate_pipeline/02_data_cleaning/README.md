# 02 - Data Cleaning

This notebook prepares the waitlist data for modeling.

The original dataframe was preserved and the cleaning steps were performed on a separate copy.

The main cleaning work included date conversion, checking invalid timelines, handling unrealistic BMI values, cleaning functional status, handling CPRA, and creating safe survival-time fields.

The cleaning decisions are recorded in `outputs/medimate_cleaning_log.csv`.

The cleaned patient-level datasets are not included in GitHub.
