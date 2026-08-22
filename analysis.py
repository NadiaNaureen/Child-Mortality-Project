"""Reproducible preparation and modelling for the Child Mortality dashboard."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent / "child mortality dataset"
KEYS = ["Entity", "Code", "Year"]
TARGET = "under5_mortality_per_1000"
FEATURES = ["health_spend_ppp", "female_school_years", "poverty_share", "gdp_per_capita_ppp"]
CLUSTER_LABELS = {0: "Transition / mixed", 1: "Infectious & hygiene-heavy", 2: "Neonatal & congenital-heavy"}


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def _numeric_column(frame: pd.DataFrame, terms: list[str], exclude: tuple[str, ...] = ()) -> str:
    candidates = [c for c in frame.columns if all(t.lower() in c.lower() for t in terms)
                  and not any(x.lower() in c.lower() for x in exclude)]
    if not candidates:
        raise KeyError(f"No column matching {terms}")
    return candidates[0]


def _indicator(name: str, output: str, terms: list[str], exclude: tuple[str, ...] = ()) -> pd.DataFrame:
    frame = _read(name)
    value = _numeric_column(frame, terms, exclude)
    return frame[KEYS + [value]].rename(columns={value: output}).drop_duplicates(KEYS)


def build_panel() -> pd.DataFrame:
    """Merge source tables into a country-year panel and interpolate within country only.

    Interpolation is deliberately performed after merging, and only across years for
    the same country. It never fills values from a different country.
    """
    mortality = _indicator("child-mortality-igme.csv", TARGET, ["Mortality rate", "under-5"])
    inputs = [
        _indicator("per-capita-total-expenditure-on-health-vs-child-mortality.csv", "health_spend_ppp", ["health expenditure", "per capita"]),
        _indicator("correlation-between-child-mortality-and-mean-years-of-schooling-for-those-aged-15-and-older.csv", "female_school_years", ["Mean years", "schooling"]),
        _indicator("poverty-and-child-mortality.csv", "poverty_share", ["share", "poverty"]),
        _indicator("diarrheal-death-rates-children-vs-gdp-per-capita.csv", "gdp_per_capita_ppp", ["GDP per capita"]),
        _indicator("children-sleeping-under-treated-bednet.csv", "bednet_coverage", ["bed nets"]),
        _indicator("exclusive-breastfeeding.csv", "exclusive_breastfeeding", ["Exclusive breastfeeding"]),
        _indicator("pneumonia-careseeking.csv", "pneumonia_careseeking", ["Percentage", "pneumonia"]),
    ]
    panel = mortality.copy()
    for source in inputs:
        panel = panel.merge(source, on=KEYS, how="left")
    panel = panel.dropna(subset=["Code", TARGET]).sort_values(["Code", "Year"])
    numeric = [c for c in panel.columns if c not in ["Entity", "Code", "Year"]]
    panel[numeric] = panel.groupby("Code")[numeric].transform(
        lambda g: g.interpolate(limit_direction="both")
    )
    return panel.reset_index(drop=True)


def build_clusters() -> tuple[pd.DataFrame, list[str]]:
    raw = _read("causes-of-death-in-children.csv").dropna(subset=["Code"])
    cause_cols = [c for c in raw.columns if c not in KEYS]
    latest = raw.sort_values("Year").groupby("Code", as_index=False).tail(1).copy()
    latest["total_cause_deaths"] = latest[cause_cols].sum(axis=1, min_count=1)
    latest = latest[latest["total_cause_deaths"] > 0].copy()
    groups = {
        "infectious_share": ["Malaria", "HIV/AIDS", "Meningitis", "Nutritional", "Whooping", "Lower respiratory", "Measles", "sepsis", "Tuberculosis", "Diarrheal", "Syphilis"],
        "neonatal_share": ["Other neonatal", "Congenital", "encephalopathy", "preterm"],
    }
    for group, terms in groups.items():
        cols = [c for c in cause_cols if any(t.lower() in c.lower() for t in terms)]
        latest[group] = latest[cols].sum(axis=1) / latest["total_cause_deaths"]
    latest["other_share"] = (1 - latest["infectious_share"] - latest["neonatal_share"]).clip(lower=0)
    shares = ["infectious_share", "neonatal_share", "other_share"]
    latest[shares] = latest[shares].replace([np.inf, -np.inf], np.nan)
    usable = latest.dropna(subset=shares).copy()
    km = KMeans(n_clusters=3, random_state=42, n_init=25)
    usable["cluster_id"] = km.fit_predict(StandardScaler().fit_transform(usable[shares]))
    # Re-label clusters by their clinical composition, not arbitrary KMeans identifiers.
    centers = usable.groupby("cluster_id")[shares].mean()
    infectious = centers["infectious_share"].idxmax()
    neonatal = centers.drop(index=infectious)["neonatal_share"].idxmax()
    transition = next(i for i in centers.index if i not in {infectious, neonatal})
    labels = {infectious: "Infectious & hygiene-heavy", neonatal: "Neonatal & congenital-heavy", transition: "Transition / mixed"}
    usable["cluster"] = usable["cluster_id"].map(labels)
    return usable[["Entity", "Code", "Year", "cluster", *shares, "total_cause_deaths"]], shares


def train_models(panel: pd.DataFrame):
    sample = panel.dropna(subset=[TARGET]).copy()
    sample = sample[sample[FEATURES].notna().sum(axis=1) >= 2]
    X, y = sample[FEATURES], sample[TARGET]
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("rf", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1))])
    model.fit(X, y)
    pred = model.predict(X)
    importances = pd.DataFrame({"feature": FEATURES, "importance": model.named_steps["rf"].feature_importances_}).sort_values("importance", ascending=False)
    metrics = {"rows": len(sample), "rf_mae_in_sample": mean_absolute_error(y, pred), "rf_r2_in_sample": r2_score(y, pred)}
    ebm = None
    try:
        from interpret.glassbox import ExplainableBoostingRegressor
        ebm = Pipeline([("imputer", SimpleImputer(strategy="median")), ("ebm", ExplainableBoostingRegressor(random_state=42, interactions=0, max_rounds=5000))])
        ebm.fit(X, y)
        ebm_pred = ebm.predict(X)
        metrics["ebm_mae_in_sample"] = mean_absolute_error(y, ebm_pred)
        metrics["ebm_r2_in_sample"] = r2_score(y, ebm_pred)
    except ImportError:
        pass
    # Country-level backsliding classifier: positive if recent mortality slope is non-negative.
    status = forecast_sdg(panel)
    latest = panel.sort_values("Year").groupby("Code", as_index=False).tail(1).merge(status[["Code", "backsliding"]], on="Code", how="inner")
    eligible = latest.dropna(subset=FEATURES + ["backsliding"])
    churn = None
    if len(eligible) >= 20 and eligible["backsliding"].nunique() == 2:
        churn = Pipeline([("imputer", SimpleImputer(strategy="median")), ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))])
        churn.fit(eligible[FEATURES], eligible["backsliding"])
    return model, importances, metrics, churn, ebm


def forecast_sdg(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, group in panel.dropna(subset=[TARGET]).groupby("Code"):
        group = group.sort_values("Year").tail(10)
        if len(group) < 3:
            continue
        lr = LinearRegression().fit(group[["Year"]], group[TARGET])
        forecast = max(0, float(lr.predict(pd.DataFrame({"Year": [2030]}))[0]))
        slope = float(lr.coef_[0])
        rows.append({"Code": code, "Entity": group["Entity"].iloc[-1], "latest_year": int(group["Year"].iloc[-1]), "latest_mortality": float(group[TARGET].iloc[-1]), "annual_change": slope, "forecast_2030": forecast, "sdg_status": "On track" if forecast <= 25 else "Off track", "backsliding": int(slope >= 0)})
    return pd.DataFrame(rows)


def intervention_costs(cluster: str, lives_saved_per_100k: float) -> pd.DataFrame:
    """Illustrative CEA using the unit costs specified in the project brief."""
    choices = [("Clean water access", 2_000_000), ("Bed nets", 500_000), ("Female schooling", 7_500_000)]
    if "Neonatal" in cluster:
        choices = [("Advanced neonatal care (proxy)", 10_000_000), *choices]
    rows = []
    for name, cost in choices:
        rows.append({"intervention": name, "illustrative_cost_usd_per_100k": cost, "lives_saved_per_100k": lives_saved_per_100k, "cost_per_life_saved_usd": cost / max(lives_saved_per_100k, 0.1)})
    return pd.DataFrame(rows)
