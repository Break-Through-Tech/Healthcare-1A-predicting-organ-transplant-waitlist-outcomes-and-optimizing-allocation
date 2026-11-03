# 05 - Survival Modeling

This notebook studies how risk changes over time while a patient is on the kidney waitlist.

Transplant is an important competing event, so the analysis uses competing-risk estimates instead of treating every transplant as ordinary censoring.

I compared an interpretable Cox model with an XGBoost survival model.

The final XGBoost survival model reached a 2024 test C-index of 0.6784.

For the one-year risk groups, estimated adverse incidence was about:

- Low risk: 1.70%
- Medium risk: 3.71%
- High risk: 6.60%

The high-risk group therefore had about 3.87 times the one-year adverse incidence of the low-risk group.

The outputs include competing-risk summaries, Cox hazard ratios, model comparison results, and survival feature importance.
