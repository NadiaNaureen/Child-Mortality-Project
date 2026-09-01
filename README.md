# Global Child Mortality Dashboard

An end-to-end Streamlit dashboard built from the supplied child-mortality CSVs and the project brief.

## What it does

- Reconstructs a country-year panel by joining mortality, health expenditure, female education, poverty, GDP, bed-net, breastfeeding and pneumonia-care indicators on `Code + Year`.
- Applies country-only time interpolation for numeric indicators, preserving the source data's geographic boundaries.
- Clusters countries by normalized cause-of-death composition (infectious, neonatal/congenital, other), not wealth.
- Uses an Explainable Boosting Machine (EBM) as the forecasting and decision model, evaluated on countries withheld from training.
- Uses the previous observed country mortality rate plus socioeconomic indicators for one-year-ahead EBM forecasting; reports EBM local contributions and optional EBM SHAP/LIME explanations.
- Forecasts each country’s 2030 under-five mortality rate from its most recent ten-year trajectory and compares it with SDG 3.2 (≤25 deaths/1,000 live births).
- Includes an interactive counterfactual policy simulator and illustrative cost-effectiveness table.

## Run it

Use Python 3.10+ in the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Optional EBM, SHAP, and LIME panels (the dashboard works without them):

```powershell
pip install --timeout 180 --retries 5 -r requirements-advanced.txt
```

The browser opens at `http://localhost:8501`. Pick a country and reference year in the sidebar, then move across the tabs:

1. **Overview**: the integrated-panel coverage, country trend and GDP/mortality relationship.
2. **Clinical clusters**: ternary cause-of-death view and cluster-specific interventions.
3. **Predict & explain**: estimated mortality, Random Forest feature importance and local SHAP contributions.
4. **SDG 2030**: country forecasts, on-track/off-track classification and backsliding flag.
5. **Policy simulator**: change schooling, health expenditure and poverty assumptions; compare the model's baseline and scenario estimate, then inspect the illustrative cost per life saved.

## Interpretation boundaries

The models quantify associations in historical, incomplete cross-country data. They do not establish causality, substitute for surveillance, or provide clinical recommendations. Forecasts extrapolate recent linear trends and should be treated as a transparent baseline, not a certainty.
