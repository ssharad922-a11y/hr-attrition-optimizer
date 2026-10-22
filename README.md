# Prescriptive HR Attrition & Retention Budget Optimizer

An end-to-end People Analytics platform coupling **calibrated machine learning**, **explainable AI (TreeSHAP)**, and **operations research (Mixed-Integer Linear Programming)** to prescribe targeted interventions and optimize retention budgets under constrained enterprise resources.

---

## Key Features

- **Collinearity Mitigation & Feature Engineering**: Addressed extreme variance inflation across employee tenure variables by constructing career progression velocity metrics (`StagnationRatio` and `ManagerTenureRatio`).
- **Probabilistic Calibration**: Benchmarked Logistic Regression against LightGBM; applied isotonic/sigmoid calibration via `CalibratedClassifierCV` to reduce Brier score loss to ~0.10, ensuring risk outputs represent empirical probabilities.
- **Explainable Root-Cause Attribution**: Deployed a SHAP `TreeExplainer` to decompose predicted probabilities into directional drivers—identifying non-cash burnout catalysts (`OverTime`) and retention anchors (`StockOptions`).
- **Financial Replacement Costing**: Integrated standard 1.5x annual compensation replacement multipliers to calculate financial value-at-risk.
- **Constrained Optimization (0/1 Knapsack)**: Solved budget allocation via `scipy.optimize.milp` to maximize turnover loss prevented under a fixed corporate budget cap ($25,000), achieving >3,000% portfolio ROI.
- **Interactive Streamlit UI**: Fully modular executive diagnostic dashboard with individual flight risk breakdowns and dynamic budget allocation controls.

---

## Directory Structure

```text
├── apps.py                  # Interactive Streamlit application
├── preprocessor.pkl         # Fitted Scikit-Learn ColumnTransformer
├── calibrated_model.pkl     # Calibrated LightGBM model for scoring
├── lgbm_booster.pkl         # Base LightGBM booster for SHAP TreeExplainer
├── test_data.csv            # Test set cohort for diagnostics
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation