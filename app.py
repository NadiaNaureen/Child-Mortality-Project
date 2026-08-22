from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import FEATURES, TARGET, build_clusters, build_panel, forecast_sdg, intervention_costs, train_models

st.set_page_config(page_title="Global Child Mortality", page_icon="🌍", layout="wide")

@st.cache_data(show_spinner="Integrating country-year data…")
def load_data():
    panel = build_panel()
    clusters, shares = build_clusters()
    return panel, clusters, shares

@st.cache_resource(show_spinner="Training analytical models…")
def load_models():
    panel, _, _ = load_data()
    return train_models(panel)

panel, clusters, shares = load_data()
model, importances, metrics, churn_model, ebm_model = load_models()
forecast = forecast_sdg(panel)

st.title("🌍 Global Child Mortality Analysis & Policy Simulator")
st.caption("Integrated country-year evidence • clinical-profile segmentation • explainable prediction • SDG 3.2 trajectory assessment")

with st.sidebar:
    st.header("Analysis filters")
    years = sorted(panel.Year.dropna().astype(int).unique())
    selected_year = st.select_slider("Reference year", options=years, value=years[-1])
    countries = sorted(panel.Entity.dropna().unique())
    country = st.selectbox("Country for detailed analysis", countries, index=countries.index("India") if "India" in countries else 0)

latest = panel[panel.Year == selected_year].copy()
latest = latest.merge(clusters[["Code", "cluster"]], on="Code", how="left")
country_data = panel[panel.Entity == country].sort_values("Year")
latest_country = country_data.iloc[-1]
country_cluster = clusters.loc[clusters.Code == latest_country.Code, "cluster"]
country_cluster = country_cluster.iloc[0] if len(country_cluster) else "Unclassified"

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Clinical clusters", "Predict & explain", "SDG 2030", "Policy simulator"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Countries / areas", f"{panel.Code.nunique():,}")
    c2.metric("Integrated panel rows", f"{len(panel):,}")
    c3.metric("Latest median U5MR", f"{latest[TARGET].median():.1f} per 1,000")
    c4.metric("Random Forest fit (in-sample R²)", f"{metrics['rf_r2_in_sample']:.2f}", help="Exploratory fit; do not treat as causal evidence.")
    st.subheader(f"Mortality trend: {country}")
    st.plotly_chart(px.line(country_data, x="Year", y=TARGET, markers=True, labels={TARGET: "Under-5 deaths per 1,000 live births"}), use_container_width=True)
    st.subheader("Socio-economic association at selected year")
    scatter = latest.dropna(subset=["gdp_per_capita_ppp", TARGET])
    st.plotly_chart(px.scatter(scatter, x="gdp_per_capita_ppp", y=TARGET, color="cluster", hover_name="Entity", log_x=True, labels={"gdp_per_capita_ppp": "GDP per capita, PPP", TARGET: "Under-5 mortality / 1,000"}), use_container_width=True)
    with st.expander("Data preparation and safeguards"):
        st.write("The dashboard merges data by country code and year. Numeric indicators are interpolated only within the same country; no value is borrowed from another country. Sparse indicators remain missing where a country has no time coverage.")
        st.markdown("""
| Source file | Role in the dashboard |
|---|---|
| `child-mortality-igme.csv` | Target: under-five mortality rate |
| `per-capita-total-expenditure-on-health-vs-child-mortality.csv` | Health spending per capita |
| `correlation-between-child-mortality-and-mean-years-of-schooling-for-those-aged-15-and-older.csv` | Female schooling years |
| `poverty-and-child-mortality.csv` | Extreme-poverty share |
| `diarrheal-death-rates-children-vs-gdp-per-capita.csv` | GDP per capita, PPP |
| `children-sleeping-under-treated-bednet.csv` | Bed-net coverage (context) |
| `exclusive-breastfeeding.csv` | Breastfeeding coverage (context) |
| `pneumonia-careseeking.csv` | Pneumonia care-seeking (context) |
| `causes-of-death-in-children.csv` | Cause-of-death clustering |

**Preprocessing:** deduplicate country-year rows; join on `Code + Year`; discard rows without valid target mortality; sort chronologically; interpolate numeric values within each country only; retain unresolved gaps; median-impute model inputs; convert cause deaths to shares; and standardize shares before K-Means. No values are transferred across countries.
        """)

with tab2:
    st.subheader("Clinical-profile clustering")
    st.write("K-Means (k=3) is applied to the latest available country cause-of-death composition, expressed as shares of recorded under-5 cause deaths—not income or GDP.")
    fig = px.scatter_ternary(clusters, a="infectious_share", b="neonatal_share", c="other_share", color="cluster", hover_name="Entity", size="total_cause_deaths", size_max=16, labels={"infectious_share":"Infectious", "neonatal_share":"Neonatal/congenital", "other_share":"Other"})
    st.plotly_chart(fig, use_container_width=True)
    summary = clusters.groupby("cluster")[shares + ["total_cause_deaths"]].mean().reset_index()
    st.dataframe(summary.style.format({c: "{:.1%}" for c in shares} | {"total_cause_deaths":"{:,.0f}"}), use_container_width=True)
    st.info({"Infectious & hygiene-heavy":"Prioritize clean water, hygiene, bed nets, nutrition and immunisation.", "Transition / mixed":"Strengthen prenatal care, rural neonatal units and maternal nutrition.", "Neonatal & congenital-heavy":"Focus on maternal-fetal medicine, NICUs and preterm prevention."}.get(country_cluster, "No clinical classification is available for this country."))

with tab3:
    st.subheader(f"Explainable mortality estimate: {country}")
    record = latest_country[FEATURES].to_frame().T
    predicted = float(model.predict(record)[0])
    st.metric("Predicted under-5 mortality", f"{predicted:.1f} per 1,000", f"Observed latest: {latest_country[TARGET]:.1f}" if pd.notna(latest_country[TARGET]) else None)
    st.plotly_chart(px.bar(importances, x="importance", y="feature", orientation="h", title="Global model importance (Random Forest)"), use_container_width=True)
    st.caption("This is a model explanation (predictive association), not proof that changing one factor causes the shown outcome.")
    shap_tab, lime_tab, ebm_tab = st.tabs(["SHAP (Random Forest)", "LIME (Random Forest)", "EBM"])
    with shap_tab:
        try:
            import shap
            imputed = model.named_steps["imputer"].transform(record)
            explainer = shap.TreeExplainer(model.named_steps["rf"])
            values = np.asarray(explainer.shap_values(imputed)).reshape(-1)
            local = pd.DataFrame({"feature": FEATURES, "SHAP contribution": values}).sort_values("SHAP contribution")
            st.plotly_chart(px.bar(local, x="SHAP contribution", y="feature", orientation="h", color="SHAP contribution", color_continuous_scale="RdBu_r", title="Local SHAP contributions"), use_container_width=True)
        except ImportError:
            st.info("Install optional packages using `pip install -r requirements-advanced.txt` to enable SHAP.")
        except Exception as exc:
            st.warning(f"SHAP explanation could not be computed: {exc}")
    with lime_tab:
        try:
            from lime.lime_tabular import LimeTabularExplainer
            training = model.named_steps["imputer"].transform(panel[FEATURES])
            explainer = LimeTabularExplainer(training, feature_names=FEATURES, mode="regression", random_state=42)
            point = model.named_steps["imputer"].transform(record)[0]
            explanation = explainer.explain_instance(point, model.named_steps["rf"].predict, num_features=len(FEATURES))
            lime_df = pd.DataFrame(explanation.as_list(), columns=["feature interval", "local weight"])
            st.plotly_chart(px.bar(lime_df, x="local weight", y="feature interval", orientation="h", color="local weight", color_continuous_scale="RdBu_r", title="LIME local approximation"), use_container_width=True)
        except ImportError:
            st.info("Install optional packages using `pip install -r requirements-advanced.txt` to enable LIME.")
        except Exception as exc:
            st.warning(f"LIME explanation could not be computed: {exc}")
    with ebm_tab:
        if ebm_model is None:
            st.info("Install optional packages using `pip install -r requirements-advanced.txt`, then restart Streamlit to train the EBM.")
        else:
            ebm_prediction = float(ebm_model.predict(record)[0])
            st.metric("EBM predicted under-5 mortality", f"{ebm_prediction:.1f} per 1,000", f"EBM in-sample R²: {metrics['ebm_r2_in_sample']:.2f}")
            try:
                point = ebm_model.named_steps["imputer"].transform(record)
                local = ebm_model.named_steps["ebm"].explain_local(point).data(0)
                contributions = pd.DataFrame({"feature": local["names"], "EBM contribution": local["scores"]}).sort_values("EBM contribution")
                st.plotly_chart(px.bar(contributions, x="EBM contribution", y="feature", orientation="h", color="EBM contribution", color_continuous_scale="RdBu_r", title="EBM local additive contributions"), use_container_width=True)
            except Exception as exc:
                st.warning(f"EBM local explanation could not be rendered: {exc}")

with tab4:
    st.subheader("SDG 3.2 trajectory forecast")
    st.write("A country-specific linear trend over its most recent ten observations forecasts the 2030 under-5 mortality rate. SDG 3.2 is assessed against ≤25 deaths per 1,000 live births.")
    c1, c2, c3 = st.columns(3)
    c1.metric("On track", int((forecast.sdg_status == "On track").sum()))
    c2.metric("Off track", int((forecast.sdg_status == "Off track").sum()))
    c3.metric("Backsliding trajectories", int(forecast.backsliding.sum()))
    st.plotly_chart(px.scatter(forecast, x="latest_mortality", y="forecast_2030", color="sdg_status", hover_name="Entity", hover_data=["annual_change", "latest_year"], labels={"latest_mortality":"Latest U5MR", "forecast_2030":"Forecast U5MR (2030)"}).add_hline(y=25, line_dash="dash", annotation_text="SDG threshold"), use_container_width=True)
    country_forecast = forecast[forecast.Entity == country]
    if not country_forecast.empty:
        st.dataframe(country_forecast.drop(columns="Code").style.format({"latest_mortality":"{:.1f}", "annual_change":"{:.2f}", "forecast_2030":"{:.1f}"}), use_container_width=True)
    if churn_model is not None:
        risk_input = latest_country[FEATURES].to_frame().T
        probability = float(churn_model.predict_proba(risk_input)[0, 1])
        st.caption(f"Exploratory trajectory-backsliding classifier probability for {country}: {probability:.0%}. This label is based on recent non-improving mortality trends, not programme dropout records.")

with tab5:
    st.subheader(f"Policy simulator: {country}")
    st.write("Move the policy levers to create a counterfactual feature profile. Predictions are scenario estimates, not causal guarantees.")
    a, b, c = st.columns(3)
    education_gain = a.slider("Increase female schooling (years)", 0.0, 5.0, 1.0, 0.1)
    health_gain = b.slider("Increase health spending (%)", 0, 200, 25, 5)
    poverty_reduction = c.slider("Reduce poverty (percentage points)", 0.0, 30.0, 5.0, 0.5)
    scenario = record.copy()
    scenario["female_school_years"] = scenario["female_school_years"].fillna(panel.female_school_years.median()) + education_gain
    scenario["health_spend_ppp"] = scenario["health_spend_ppp"].fillna(panel.health_spend_ppp.median()) * (1 + health_gain / 100)
    scenario["poverty_share"] = (scenario["poverty_share"].fillna(panel.poverty_share.median()) - poverty_reduction).clip(lower=0)
    baseline = float(model.predict(record)[0]); scenario_prediction = float(model.predict(scenario)[0])
    saved = max(0, baseline - scenario_prediction) * 100  # per 100,000 live births
    x, y, z = st.columns(3)
    x.metric("Baseline estimate", f"{baseline:.1f} / 1,000")
    y.metric("Scenario estimate", f"{scenario_prediction:.1f} / 1,000", f"{scenario_prediction - baseline:.1f}")
    z.metric("Estimated lives saved", f"{saved:.0f} per 100k live births")
    cea = intervention_costs(country_cluster, saved)
    st.dataframe(cea.style.format({"illustrative_cost_usd_per_100k":"${:,.0f}", "lives_saved_per_100k":"{:,.1f}", "cost_per_life_saved_usd":"${:,.0f}"}), use_container_width=True)
    st.caption("Cost-effectiveness assumptions from the brief: $20/person for clean water, $5/bed net, and $150/girl-year of schooling. Costs are illustrative and should be replaced with country programme budgets before policy use.")

st.divider()
st.caption("Sources: supplied OWID/IGME/IHME/GAPMINDER-derived CSV files. This is an academic decision-support prototype, not clinical or causal policy advice.")
