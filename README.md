# Loan Default Risk & Pricing, LendingClub Dataset

**By Elahe, Ayush and Benjamin**

**Live app:** http://eab-credit.preview.james-benjamin.com/

## The Problem
A loan officer (lender) commits money to a borrower without knowing whether it will be repaid or not (defaulted).
We predict the **probability of default** using only what is known at application time.

From a business perspective, there are two errors with unequal costs:
- **Approving a defaulter** loses most of the principal
- **Rejecting a good borrower** loses only the profit on that loan

We assume the first mistake costs more to the business (5×), and this asymmetry drives how we set the decision threshold and the loan pricing.

**Research question:** *Can a loan officer, using only application-time information, predict whether a requested loan will default, and use this likelihood to decide whether to grant the loan?*

## Data
- Source: [LendingClub accepted loans 2007–2018](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (Kaggle, ~2.2 million loans)
- Finished loans only: `Fully Paid` = 0, `Charged Off` = 1 (about 20% defaults)
- Only information known at application time is used as model input. LendingClub's own grade and interest rate
  are kept only as a benchmark, and post-loan fields only to measure losses.
- `data/loans_clean.csv.gz` is the preprocessed sample with the engineered features (`loan_to_income`, credit history length)

## Notebook
`lending_notebook.ipynb`

| Part | Sections | Owner | Content |
|---|---|---|---|
| A | 1–8 | Elahe | Problem, leakage policy, cleaning, EDA, feature engineering, time-based split, preprocessing, baselines |
| B | 9–12 | Ayush | Bias–variance, Random Forest, tuned XGBoost, feature selection |
| C | 13–17 | Benjamin | Model comparison, calibration, evaluation, pricing, deployment |

## How to run
```bash
pip install "pandas<3" numpy matplotlib scikit-learn xgboost joblib kagglehub jupyter
jupyter notebook lending_notebook.ipynb
```
The dataset is downloaded automatically with `kagglehub` on the first run.
