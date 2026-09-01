from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import FEATURES, TARGET, build_clusters, build_panel, forecast_sdg, intervention_costs, train_models

st.set_page_config(page_title="Child Mortality Intelligence", layout="wide", initial_sidebar_state="expanded")

FEATURE_LABELS = {
    "prior_u5mr": "Previous-year under-five mortality",
    "health_spend_ppp": "Health spending per capita (PPP)",
    "female_school_years": "Female schooling years",
    "poverty_share": "Extreme-poverty share",
    "gdp_per_capita_ppp": "GDP per capita (PPP)",
}
PALETTE = {"Infectious & hygiene-heavy": "#E66A4E", "Transition / mixed": "#E7A53B", "Neonatal & congenital-heavy": "#305F8C"}

st.markdown("""
<style>
.stApp {background:#F7F9FC;color:#132238} [data-testid="stSidebar"] {background:#102A43}
[data-testid="stSidebar"] * {color:#F5F8FC !important}.block-container {max-width:1380px;padding-top:2.2rem}
[data-testid="stSidebar"] [data-baseweb="select"], [data-testid="stSidebar"] [data-baseweb="select"] * {background:#FFFFFF !important;color:#132238 !important}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] input:disabled {color:#132238 !important;-webkit-text-fill-color:#132238 !important;opacity:1 !important}
.hero {padding:1.35rem 1.6rem;border-radius:18px;background:linear-gradient(120deg,#102A43,#1F5F8B);color:white;margin-bottom:1.35rem}
.hero h1 {margin:0;font-size:2rem;color:white}.hero p {margin:.35rem 0 0;color:#DDEAF4}
[data-testid="stMetric"] {background:white;border:1px solid #E3EAF2;border-radius:14px;padding:.75rem 1rem}
.section-note {color:#52677D;font-size:.92rem;margin-bottom:.8rem}.primary-tag {display:inline-block;color:#0F5B41;background:#DDF4E9;border-radius:100px;padding:.2rem .6rem;font-size:.78rem;font-weight:700}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Building the integrated country-year panel...")
def load_data():
    panel = build_panel()
    clusters, shares = build_clusters()
    return panel, clusters, shares


@st.cache_resource(show_spinner="Training the EBM forecasting model...")
def load_model():
    panel, _, _ = load_data()
    return train_models(panel)


def pretty_term(term: str) -> str:
    for index, feature in enumerate(FEATURES):
        term = term.replace(f"feature_{index:04d}", FEATURE_LABELS[feature])
    return term


panel, clusters, shares = load_data()
ebm_model, metrics, churn_model = load_model()
if ebm_model is None:
    st.error("EBM is not installed. Run `pip install -r requirements-advanced.txt`, then restart the dashboard.")
    st.stop()
forecast = forecast_sdg(panel)

st.markdown("""<div class="hero"><h1>Child Mortality Intelligence</h1><p>Clinical profiles, explainable one-year-ahead forecasting, and SDG 2030 decision support</p></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Analysis controls")
    years = sorted(panel.Year.dropna().astype(int).unique())
    selected_year = st.select_slider("Reference year", options=years, value=years[-1])
    countries = sorted(panel.Entity.dropna().unique())
    country = st.selectbox("Country", countries, index=countries.index("India") if "India" in countries else 0)
    st.divider()
    st.caption("Primary model")
    st.markdown("**Explainable Boosting Machine**")
    st.caption("One-year-ahead forecast using the previous observed mortality value plus socioeconomic indicators.")

latest = panel[panel.Year == selected_year].copy().merge(clusters[["Code", "cluster"]], on="Code", how="left")
country_data = panel[panel.Entity == country].sort_values("Year")
latest_country = country_data.iloc[-1]
country_cluster_result = clusters.loc[clusters.Code == latest_country.Code, "cluster"]
country_cluster = country_cluster_result.iloc[0] if len(country_cluster_result) else "Unclassified"
record = latest_country[FEATURES].to_frame().T

overview_tab, cluster_tab, model_tab, sdg_tab, simulator_tab = st.tabs(["Overview", "Clinical profiles", "EBM insights", "SDG 2030", "Policy simulator"])

with overview_tab:
    first, second, third, fourth = st.columns(4)
    first.metric("Countries / areas", f"{panel.Code.nunique():,}")
    second.metric("Country-year records", f"{len(panel):,}")
    third.metric("Latest median U5MR", f"{latest[TARGET].median():.1f} / 1,000")
    fourth.metric("EBM held-out R²", f"{metrics['ebm_r2']:.3f}", help="Countries in the validation group were not used to fit the model.")
    left, right = st.columns([1.15, .85], gap="large")
    with left:
        st.subheader(f"Mortality trajectory — {country}")
        line = px.line(country_data, x="Year", y=TARGET, markers=True, labels={TARGET: "Under-five mortality / 1,000"}, color_discrete_sequence=["#1F5F8B"])
        line.update_layout(margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(line, use_container_width=True)
    with right:
        st.subheader(f"Selected country — {country}")
        estimate = float(ebm_model.predict(record)[0])
        st.metric("EBM one-year-ahead estimate", f"{estimate:.1f} / 1,000")
        st.metric("Latest observed mortality", f"{latest_country[TARGET]:.1f} / 1,000")
        st.metric("Held-out MAE", f"{metrics['ebm_mae']:.3f}")
        st.caption("The model uses the country’s prior observed mortality and current socioeconomic context; lower MAE and higher R² indicate better validation performance.")
    st.subheader("Socio-economic pattern at the selected year")
    scatter = latest.dropna(subset=["gdp_per_capita_ppp", TARGET])
    chart = px.scatter(scatter, x="gdp_per_capita_ppp", y=TARGET, color="cluster", color_discrete_map=PALETTE, hover_name="Entity", log_x=True, labels={"gdp_per_capita_ppp": "GDP per capita, PPP", TARGET: "Under-five mortality / 1,000"})
    chart.update_layout(margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(chart, use_container_width=True)
    with st.expander("Data lineage and preprocessing"):
        st.markdown("""| Source file | Integrated role |
|---|---|
| `child-mortality-igme.csv` | Target and prior-year mortality forecast feature |
| Health-expenditure, female-schooling, poverty and GDP CSVs | EBM forecasting features |
| Bed-net, exclusive-breastfeeding and pneumonia-careseeking CSVs | Intervention context |
| `causes-of-death-in-children.csv` | Clinical-profile clustering |

**Processing:** country-year deduplication; `Code + Year` joins; invalid-target removal; chronological ordering; within-country interpolation; a one-year mortality lag; median model imputation; cause-share normalisation; and standardisation before K-Means.""")

with cluster_tab:
    st.subheader("Clinical-profile clustering")
    st.markdown("<p class='section-note'>Countries are segmented by cause-of-death composition, not GDP or income level.</p>", unsafe_allow_html=True)
    ternary = px.scatter_ternary(clusters, a="infectious_share", b="neonatal_share", c="other_share", color="cluster", color_discrete_map=PALETTE, hover_name="Entity", size="total_cause_deaths", size_max=15, labels={"infectious_share": "Infectious", "neonatal_share": "Neonatal / congenital", "other_share": "Other"})
    ternary.update_layout(margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="white")
    st.plotly_chart(ternary, use_container_width=True)
    summary = clusters.groupby("cluster")[shares + ["total_cause_deaths"]].mean().reset_index()
    st.dataframe(summary.style.format({key: "{:.1%}" for key in shares} | {"total_cause_deaths": "{:,.0f}"}), hide_index=True, use_container_width=True)
    policies = {"Infectious & hygiene-heavy": "Water, sanitation, immunisation, bed nets and nutrition are high-priority levers.", "Transition / mixed": "Strengthen antenatal care, maternal nutrition and neonatal services.", "Neonatal & congenital-heavy": "Prioritise maternal-fetal medicine, preterm prevention and specialised neonatal care."}
    st.info(f"{country}: {country_cluster}. {policies.get(country_cluster, 'No clinical classification is available.')}")

with model_tab:
    st.markdown("<span class='primary-tag'>PRIMARY FORECAST MODEL</span>", unsafe_allow_html=True)
    st.subheader(f"Explainable Boosting Machine — {country}")
    a, b, c = st.columns(3)
    a.metric("EBM estimate", f"{float(ebm_model.predict(record)[0]):.1f} / 1,000")
    b.metric("Observed latest value", f"{latest_country[TARGET]:.1f} / 1,000")
    c.metric("EBM held-out R²", f"{metrics['ebm_r2']:.3f}")
    st.markdown("<p class='section-note'>EBM is a glass-box boosting model: every prediction is built from visible feature contributions and selected interactions.</p>", unsafe_allow_html=True)
    point = ebm_model.named_steps["imputer"].transform(record)
    local = ebm_model.named_steps["ebm"].explain_local(point).data(0)
    local_ebm = pd.DataFrame({"feature": [pretty_term(name) for name in local["names"]], "EBM contribution": local["scores"]}).sort_values("EBM contribution")
    local_chart = px.bar(local_ebm, x="EBM contribution", y="feature", orientation="h", color="EBM contribution", color_continuous_scale="RdBu_r", title="Why EBM produced this country estimate")
    local_chart.update_layout(margin=dict(l=0, r=0, t=45, b=0), paper_bgcolor="white", plot_bgcolor="white", coloraxis_showscale=False)
    st.plotly_chart(local_chart, use_container_width=True)
    shap_tab, lime_tab = st.tabs(["SHAP explanation", "LIME explanation"])
    with shap_tab:
        if st.checkbox("Generate SHAP explanation", key="show_shap"):
            try:
                import shap
                background = ebm_model.named_steps["imputer"].transform(panel[FEATURES].head(80))
                explainer = shap.Explainer(ebm_model.named_steps["ebm"].predict, background)
                values = np.asarray(explainer(point).values).reshape(-1)
                shap_data = pd.DataFrame({"feature": [FEATURE_LABELS[key] for key in FEATURES], "SHAP contribution": values}).sort_values("SHAP contribution")
                st.plotly_chart(px.bar(shap_data, x="SHAP contribution", y="feature", orientation="h", color="SHAP contribution", color_continuous_scale="RdBu_r", title="EBM local SHAP explanation"), use_container_width=True)
            except ImportError:
                st.info("Install requirements-advanced.txt to enable SHAP.")
            except Exception as exc:
                st.warning(f"SHAP explanation could not be computed: {exc}")
        else:
            st.caption("Enable this on demand; SHAP is intentionally deferred so country changes remain fast.")
    with lime_tab:
        if st.checkbox("Generate LIME explanation", key="show_lime"):
            try:
                from lime.lime_tabular import LimeTabularExplainer
                training = ebm_model.named_steps["imputer"].transform(panel[FEATURES])
                explainer = LimeTabularExplainer(training, feature_names=[FEATURE_LABELS[key] for key in FEATURES], mode="regression", random_state=42)
                lime_data = pd.DataFrame(explainer.explain_instance(point[0], ebm_model.named_steps["ebm"].predict, num_features=len(FEATURES)).as_list(), columns=["feature range", "local weight"])
                st.plotly_chart(px.bar(lime_data, x="local weight", y="feature range", orientation="h", color="local weight", color_continuous_scale="RdBu_r", title="EBM local LIME explanation"), use_container_width=True)
            except ImportError:
                st.info("Install requirements-advanced.txt to enable LIME.")
            except Exception as exc:
                st.warning(f"LIME explanation could not be computed: {exc}")
        else:
            st.caption("Enable this on demand; LIME is intentionally deferred so country changes remain fast.")

with sdg_tab:
    st.subheader("SDG 3.2 trajectory forecast")
    st.markdown("<p class='section-note'>A transparent trend over each country’s latest ten observations projects 2030 mortality. The SDG reference threshold is 25 deaths per 1,000 live births.</p>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("On track", int((forecast.sdg_status == "On track").sum()))
    b.metric("Off track", int((forecast.sdg_status == "Off track").sum()))
    c.metric("Backsliding trajectories", int(forecast.backsliding.sum()))
    sdg_plot = px.scatter(forecast, x="latest_mortality", y="forecast_2030", color="sdg_status", hover_name="Entity", hover_data=["annual_change", "latest_year"], color_discrete_map={"On track": "#25876A", "Off track": "#D05A4D"}, labels={"latest_mortality": "Latest U5MR", "forecast_2030": "Forecast U5MR (2030)"})
    sdg_plot.add_hline(y=25, line_dash="dash", annotation_text="SDG threshold")
    sdg_plot.update_layout(margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(sdg_plot, use_container_width=True)
    country_forecast = forecast[forecast.Entity == country]
    if not country_forecast.empty:
        st.dataframe(country_forecast.drop(columns="Code").style.format({"latest_mortality": "{:.1f}", "annual_change": "{:.2f}", "forecast_2030": "{:.1f}"}), hide_index=True, use_container_width=True)

with simulator_tab:
    st.subheader(f"Policy scenario simulator — {country}")
    st.markdown("<p class='section-note'>Change policy inputs and compare baseline and scenario EBM estimates. Results are model scenarios, not causal guarantees.</p>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    education_gain = a.slider("Additional female schooling (years)", 0.0, 5.0, 1.0, 0.1)
    health_gain = b.slider("Health-spending increase (%)", 0, 200, 25, 5)
    poverty_reduction = c.slider("Poverty reduction (percentage points)", 0.0, 30.0, 5.0, 0.5)
    scenario = record.copy()
    scenario["female_school_years"] = scenario["female_school_years"].fillna(panel.female_school_years.median()) + education_gain
    scenario["health_spend_ppp"] = scenario["health_spend_ppp"].fillna(panel.health_spend_ppp.median()) * (1 + health_gain / 100)
    scenario["poverty_share"] = (scenario["poverty_share"].fillna(panel.poverty_share.median()) - poverty_reduction).clip(lower=0)
    baseline = float(ebm_model.predict(record)[0])
    scenario_prediction = float(ebm_model.predict(scenario)[0])
    net_lives = (baseline - scenario_prediction) * 100
    saved = max(0, net_lives)
    x, y, z = st.columns(3)
    x.metric("EBM baseline", f"{baseline:.1f} / 1,000")
    y.metric("EBM scenario", f"{scenario_prediction:.3f} / 1,000", f"{scenario_prediction - baseline:+.3f}")
    if net_lives > 0:
        z.metric("Estimated lives saved", f"{net_lives:.1f} per 100k live births")
    else:
        z.metric("Model-estimated additional deaths", f"{abs(net_lives):.1f} per 100k live births")
    cost_table = intervention_costs(country_cluster, saved)
    st.dataframe(cost_table.style.format({"illustrative_cost_usd_per_100k": "${:,.0f}", "lives_saved_per_100k": "{:,.1f}", "cost_per_life_saved_usd": lambda value: "N/A" if pd.isna(value) else f"${value:,.0f}"}), hide_index=True, use_container_width=True)
    if net_lives <= 0:
        st.warning("This scenario does not produce model-estimated lives saved. The EBM is showing an association in the historical data, not a causal policy effect; use this as a signal to inspect the selected inputs rather than as a recommendation.")
    st.caption("Illustrative costs: $20/person for clean water, $5/bed net and $150/girl-year of schooling. Use programme-specific budgets for policy work.")

st.divider()
st.caption("Supplied OWID / IGME / IHME / Gapminder-derived CSVs. Academic decision-support prototype; predictions describe patterns, not causal effects or clinical advice.")
