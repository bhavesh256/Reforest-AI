"""
Reforestation ML Pipeline — simplified, leakage-aware edition
=============================================================

Reads the six Global Forest Watch Excel workbooks from ./Dataset, engineers
state-level indicators, scores reforestation priorities, evaluates a small set
of widely used regression models, creates simple scenario forecasts, and saves
JSON/CSV/model outputs.

Dependencies:
    pip install pandas numpy scikit-learn openpyxl joblib

Optional:
    pip install imbalanced-learn  # not required; SMOTE intentionally omitted

Important:
- The regression target is total_loss_ha across 2001–2020.
- Predictors derived from that same loss series are excluded from regression.
- CV metrics are estimates, not a promise of future accuracy. Small state-level
  samples can produce low or negative R² without indicating a coding error.
- Forecasts are illustrative scenarios, not causal estimates of policy impact.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

warnings.filterwarnings("default", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("Dataset")
OUTPUT_DIR = Path("ml_outputs")
MODEL_DIR = Path("models")
THRESHOLD = 30
PAST_YEARS = list(range(2001, 2021))
FORECAST_YEARS = list(range(2021, 2031))
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def json_safe(value: Any) -> Any:
    """Convert NumPy/Pandas values to JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 1. Data loading / joining
# ---------------------------------------------------------------------------
class DataLoader:
    FILES = {
        "country_tcl": "Country tree cover loss.xlsx",
        "country_carbon": "Country carbon data.xlsx",
        "state_tcl": "Subnational 1 tree cover loss.xlsx",
        "state_carbon": "Subnational 1 carbon data.xlsx",
        "district_tcl": "Subnational 2 tree cover loss.xlsx",
        "district_carbon": "Subnational 2 carbon data.xlsx",
    }

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data: dict[str, pd.DataFrame] = {}

    def load_all(self) -> None:
        missing = [name for name in self.FILES.values()
                   if not (self.data_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing workbook(s) in {self.data_dir.resolve()}: {missing}"
            )
        for key, filename in self.FILES.items():
            self.data[key] = pd.read_excel(self.data_dir / filename)
            print(f"Loaded {filename}: {self.data[key].shape}")

    @staticmethod
    def _threshold_filter(df: pd.DataFrame, column: str) -> pd.DataFrame:
        if column not in df.columns:
            raise KeyError(f"Required threshold column {column!r} not found.")
        values = pd.to_numeric(df[column], errors="coerce")
        return df.loc[values.eq(THRESHOLD)].copy()

    def get_state_data(self) -> pd.DataFrame:
        tcl = self._threshold_filter(self.data["state_tcl"], "threshold")
        carbon = self._threshold_filter(
            self.data["state_carbon"], "umd_tree_cover_density__threshold"
        ).drop(columns=["umd_tree_cover_density__threshold"], errors="ignore")
        keys = ["country", "subnational1"]
        for key in keys:
            if key not in tcl or key not in carbon:
                raise KeyError(f"Expected merge key {key!r} in state workbooks.")
        merged = tcl.merge(carbon, on=keys, how="left", suffixes=("", "_carbon"))
        merged = merged.dropna(subset=["subnational1"])
        # Prevent silent row multiplication caused by duplicate join keys.
        if merged.duplicated(keys).any():
            duplicates = merged.loc[merged.duplicated(keys, keep=False), keys]
            raise ValueError(
                "Duplicate state keys after merging carbon data. Check source "
                f"workbooks. Examples: {duplicates.head().to_dict('records')}"
            )
        print(f"Merged state data: {merged.shape}")
        return merged

    def get_country_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        tcl = self._threshold_filter(self.data["country_tcl"], "threshold")
        carbon = self._threshold_filter(
            self.data["country_carbon"], "umd_tree_cover_density__threshold"
        )
        return tcl, carbon

    def get_district_data(self) -> pd.DataFrame:
        tcl = self._threshold_filter(self.data["district_tcl"], "threshold")
        carbon = self._threshold_filter(
            self.data["district_carbon"], "umd_tree_cover_density__threshold"
        ).drop(columns=["umd_tree_cover_density__threshold"], errors="ignore")
        keys = ["country", "subnational1", "subnational2"]
        merged = tcl.merge(carbon, on=keys, how="left", suffixes=("", "_carbon"))
        return merged.dropna(subset=["subnational2"])


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
class FeatureEngineer:
    def __init__(self):
        self.loss_cols = [f"tc_loss_ha_{y}" for y in PAST_YEARS]
        self.emission_cols = [
            f"gfw_gross_emissions_co2e_all_gases_{y}__Mg" for y in PAST_YEARS
        ]

    def engineer_state_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        missing_loss = [c for c in self.loss_cols if c not in result.columns]
        if missing_loss:
            raise KeyError(f"Missing annual loss columns: {missing_loss}")

        for c in self.loss_cols:
            result[c] = pd.to_numeric(result[c], errors="coerce")
        # Missing yearly observations remain missing during feature calculations;
        # sum/mean skip missing values but a row with all missing years is flagged.
        result["observed_loss_years"] = result[self.loss_cols].notna().sum(axis=1)
        result["total_loss_ha"] = result[self.loss_cols].sum(axis=1, min_count=1)
        result["avg_annual_loss_ha"] = result[self.loss_cols].mean(axis=1)
        result["max_annual_loss_ha"] = result[self.loss_cols].max(axis=1)
        result["min_annual_loss_ha"] = result[self.loss_cols].min(axis=1)
        result["std_annual_loss_ha"] = result[self.loss_cols].std(axis=1)

        for c in ["area_ha", "extent_2000_ha", "extent_2010_ha",
                  "gain_2000-2012_ha",
                  "avg_whrc_aboveground_biomass_2000_Mg_ha-1",
                  "gfw_net_flux_co2e__Mg_yr-1"]:
            if c in result:
                result[c] = pd.to_numeric(result[c], errors="coerce")

        area = result.get("area_ha", pd.Series(np.nan, index=result.index)).replace(0, np.nan)
        extent_2000 = result.get("extent_2000_ha", pd.Series(np.nan, index=result.index))
        extent_2010 = result.get("extent_2010_ha", pd.Series(np.nan, index=result.index))
        result["forest_cover_ratio_2000"] = (extent_2000 / area).clip(0, 1)
        result["forest_cover_ratio_2010"] = (extent_2010 / area).clip(0, 1)
        result["forest_cover_change_2000_2010"] = (
            (extent_2010 - extent_2000) / extent_2000.replace(0, np.nan)
        )

        gain = result.get("gain_2000-2012_ha", pd.Series(0.0, index=result.index)).fillna(0)
        result["remaining_forest_ha"] = (extent_2000 - result["total_loss_ha"] + gain).clip(lower=0)
        result["remaining_forest_ratio"] = (result["remaining_forest_ha"] / area).clip(0, 1)

        first_cols = [f"tc_loss_ha_{y}" for y in range(2001, 2011)]
        second_cols = [f"tc_loss_ha_{y}" for y in range(2011, 2021)]
        result["avg_loss_first_half"] = result[first_cols].mean(axis=1)
        result["avg_loss_second_half"] = result[second_cols].mean(axis=1)
        result["deforestation_acceleration"] = (
            (result["avg_loss_second_half"] - result["avg_loss_first_half"])
            / result["avg_loss_first_half"].replace(0, np.nan)
        )
        x_year = np.arange(len(PAST_YEARS), dtype=float)
        slopes = []
        for _, row in result[self.loss_cols].iterrows():
            vals = row.to_numpy(dtype=float)
            mask = np.isfinite(vals)
            slopes.append(float(np.polyfit(x_year[mask], vals[mask], 1)[0])
                          if mask.sum() >= 2 else 0.0)
        result["loss_trend_slope"] = slopes
        result["loss_cv"] = (
            result["std_annual_loss_ha"] / result["avg_annual_loss_ha"].replace(0, np.nan)
        )

        emissions_present = [c for c in self.emission_cols if c in result.columns]
        if emissions_present:
            for c in emissions_present:
                result[c] = pd.to_numeric(result[c], errors="coerce")
            result["total_emissions_Mg"] = result[emissions_present].sum(axis=1, min_count=1)
            result["avg_annual_emissions_Mg"] = result[emissions_present].mean(axis=1)
        else:
            result["total_emissions_Mg"] = np.nan
            result["avg_annual_emissions_Mg"] = np.nan

        # These carbon metrics are used for descriptive ranking only. Do not use
        # carbon_intensity as a regression predictor because it divides by loss.
        result["carbon_intensity"] = (
            result["total_emissions_Mg"] / result["total_loss_ha"].replace(0, np.nan)
        )
        result["net_carbon_flux"] = result.get(
            "gfw_net_flux_co2e__Mg_yr-1", pd.Series(np.nan, index=result.index)
        )
        result["is_carbon_source"] = (result["net_carbon_flux"] > 0).astype(int)
        result["biomass_density"] = result.get(
            "avg_whrc_aboveground_biomass_2000_Mg_ha-1",
            pd.Series(np.nan, index=result.index)
        )
        result["log_area"] = np.log1p(result.get("area_ha", pd.Series(np.nan, index=result.index)).clip(lower=0))
        result["log_total_emissions"] = np.log1p(result["total_emissions_Mg"].clip(lower=0))

        # Composite score is descriptive; it is not a ground-truth ML label.
        def minmax(s: pd.Series) -> pd.Series:
            s = s.astype(float)
            lo, hi = s.min(skipna=True), s.max(skipna=True)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                return pd.Series(0.0, index=s.index)
            return (s - lo) / (hi - lo)

        loss_norm = minmax(result["total_loss_ha"].fillna(0))
        accel_norm = minmax(result["deforestation_acceleration"].replace([np.inf, -np.inf], np.nan).fillna(0))
        carbon_norm = minmax(result["carbon_intensity"].replace([np.inf, -np.inf], np.nan).fillna(0))
        slope_norm = minmax(result["loss_trend_slope"].fillna(0))
        forest_norm = minmax(result["remaining_forest_ratio"].fillna(0))
        result["vulnerability_score"] = (
            0.30 * loss_norm + 0.20 * accel_norm + 0.15 * carbon_norm
            + 0.20 * slope_norm + 0.15 * (1 - forest_norm)
        )
        return result


# ---------------------------------------------------------------------------
# 3. Fuzzy priority score (kept simple and transparent)
# ---------------------------------------------------------------------------
class FuzzyLogicEngine:
    @staticmethod
    def compute_priority(row: pd.Series) -> dict[str, Any]:
        loss = float(row.get("total_loss_ha", 0) or 0)
        loss_scale = max(float(row.get("_max_loss_for_priority", loss) or loss), 1.0)
        loss_score = float(np.clip(loss / loss_scale, 0, 1))
        remaining = row.get("remaining_forest_ratio", np.nan)
        remaining = float(np.clip(remaining, 0, 1)) if pd.notna(remaining) else 0.5
        emissions = row.get("total_emissions_Mg", np.nan)
        emissions = float(emissions) if pd.notna(emissions) else 0.0
        # Normalize emissions using a value injected by the pipeline.
        emission_scale = max(float(row.get("_max_emissions_for_priority", emissions) or emissions), 1.0)
        carbon_score = float(np.clip(emissions / emission_scale, 0, 1))
        score = 100 * (0.45 * loss_score + 0.30 * (1 - remaining) + 0.25 * carbon_score)
        if score >= 75:
            label = "Critical"
        elif score >= 55:
            label = "High"
        elif score >= 35:
            label = "Moderate"
        else:
            label = "Low"
        return {"fuzzy_priority_score": round(score, 3), "fuzzy_priority_label": label}


# ---------------------------------------------------------------------------
# 4. Regression models and leakage-aware evaluation
# ---------------------------------------------------------------------------
def make_models() -> dict[str, Any]:
    # These are common, explainable baseline choices; less commonly used models,
    # stacking, SVR, CatBoost and XGBoost are intentionally removed.
    linear = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0)),
    ])
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=500, max_depth=5, min_samples_leaf=2,
            max_features=0.8, random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])
    gb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.04, max_depth=2,
            min_samples_leaf=3, loss="huber", random_state=RANDOM_STATE
        )),
    ])
    # A log-target wrapper can help with a nonnegative, right-skewed target.
    return {
        "Ridge": TransformedTargetRegressor(
            regressor=linear, func=np.log1p, inverse_func=np.expm1, check_inverse=False
        ),
        "Random Forest": TransformedTargetRegressor(
            regressor=rf, func=np.log1p, inverse_func=np.expm1, check_inverse=False
        ),
        "Gradient Boosting": TransformedTargetRegressor(
            regressor=gb, func=np.log1p, inverse_func=np.expm1, check_inverse=False
        ),
    }


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pred = np.maximum(np.asarray(pred, dtype=float), 0)
    denom = float(np.abs(y).sum())
    return {
        "r2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
        "mae": float(mean_absolute_error(y, pred)),
        "median_absolute_error": float(median_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "wape_pct": float(100 * np.abs(y - pred).sum() / denom) if denom else 0.0,
    }


class EnsembleModels:
    def __init__(self):
        self.models: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}
        self.feature_importance: dict[str, Any] = {}
        self.best_model_name: str | None = None

    def train_all(self, df: pd.DataFrame, feature_cols: list[str],
                  target_col: str = "total_loss_ha") -> tuple[dict, dict]:
        if not feature_cols:
            raise ValueError("No usable independent predictor columns were found.")
        data = df[feature_cols + [target_col]].copy()
        data = data.replace([np.inf, -np.inf], np.nan)
        data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
        data = data.dropna(subset=[target_col])
        data = data.loc[data[target_col] >= 0]
        X = data[feature_cols].apply(pd.to_numeric, errors="coerce")
        y = data[target_col].to_numpy(dtype=float)
        if len(y) < 6:
            raise ValueError(f"At least 6 valid state rows are required; found {len(y)}.")

        n_splits = min(5, len(y))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        self.metrics = {}
        self.models = {}
        for name, estimator in make_models().items():
            # Out-of-fold predictions: each row is predicted by a model that did
            # not train on that row. This is more informative than train R².
            oof_pred = cross_val_predict(clone(estimator), X, y, cv=cv, n_jobs=None)
            metrics = regression_metrics(y, oof_pred)
            metrics["n_samples"] = int(len(y))
            metrics["cv"] = f"{n_splits}-fold shuffled cross-validation"
            metrics["oof_predictions"] = oof_pred.tolist()
            metrics["actual_values"] = y.tolist()
            self.metrics[name] = metrics

            final_model = clone(estimator)
            final_model.fit(X, y)
            self.models[name] = final_model

            # Extract feature importances where the final estimator supports it.
            fitted_reg = getattr(final_model, "regressor_", None)
            if fitted_reg is not None and hasattr(fitted_reg, "named_steps"):
                inner = fitted_reg.named_steps.get("model")
                if hasattr(inner, "feature_importances_"):
                    self.feature_importance[name] = dict(zip(
                        feature_cols, map(float, inner.feature_importances_)
                    ))
            print(f"{name:18s} OOF R²={metrics['r2']:.3f} | "
                  f"MAE={metrics['mae']:,.1f} | RMSE={metrics['rmse']:,.1f}")

        # Select on out-of-fold MAE rather than the luckiest R².
        self.best_model_name = min(self.metrics, key=lambda n: self.metrics[n]["mae"])
        self.metrics["_selection"] = {
            "best_model_by_oof_mae": self.best_model_name,
            "note": "CV scores are estimates; no score improvement is guaranteed.",
        }
        return self.models, self.metrics


# ---------------------------------------------------------------------------
# 5. Simple transparent trend scenarios
# ---------------------------------------------------------------------------
class TimeSeriesForecaster:
    def forecast_state(self, state_name: str, annual_losses: list[float],
                       annual_emissions: list[float] | None = None) -> dict[str, Any]:
        years = np.asarray(PAST_YEARS, dtype=float)
        losses = np.asarray(annual_losses, dtype=float)
        losses[~np.isfinite(losses)] = 0
        recent = losses[-5:]
        baseline = max(float(np.mean(recent)), 0.0)
        # Fit only a linear trend; clamp physical quantities to nonnegative.
        slope = float(np.polyfit(years[-10:], losses[-10:], 1)[0]) if len(losses) >= 10 else 0.0
        future = np.asarray(FORECAST_YEARS, dtype=float)
        trend = np.maximum(baseline + slope * (future - future[0]), 0)
        bau = 0.5 * trend + 0.5 * baseline
        # This is an illustrative scenario assumption, not a learned policy effect.
        reduction_rates = np.minimum(0.05 + 0.02 * np.arange(len(future)), 0.20)
        reforest = np.maximum(bau * (1 - reduction_rates), 0)

        bau_emissions, reforest_emissions = np.zeros(len(future)), np.zeros(len(future))
        if annual_emissions is not None:
            emissions = np.asarray(annual_emissions, dtype=float)
            emissions[~np.isfinite(emissions)] = 0
            ratio = max(float(emissions.sum()) / max(float(losses.sum()), 1.0), 0.0)
            bau_emissions = bau * ratio
            reforest_emissions = reforest * ratio

        return {
            "state": state_name,
            "past": {"years": PAST_YEARS, "annual_loss_ha": losses.tolist()},
            "bau_scenario": {
                "years": FORECAST_YEARS,
                "annual_loss_ha": bau.tolist(),
                "annual_emissions_Mg": bau_emissions.tolist(),
            },
            "reforestation_scenario": {
                "years": FORECAST_YEARS,
                "annual_loss_ha": reforest.tolist(),
                "annual_emissions_Mg": reforest_emissions.tolist(),
            },
            "projected_loss_avoidance_ha": float(max(0, bau.sum() - reforest.sum())),
            "carbon_savings_Mg": float(max(0, bau_emissions.sum() - reforest_emissions.sum())),
            "trend_slope_ha_per_year": slope,
            "scenario_note": "Illustrative assumptions; not a causal impact estimate.",
        }

    def forecast_all_states(self, df: pd.DataFrame) -> dict[str, Any]:
        loss_cols = [f"tc_loss_ha_{y}" for y in PAST_YEARS]
        emission_cols = [f"gfw_gross_emissions_co2e_all_gases_{y}__Mg" for y in PAST_YEARS]
        forecasts = {}
        for _, row in df.iterrows():
            name = str(row["subnational1"])
            loss = pd.to_numeric(row[loss_cols], errors="coerce").fillna(0).tolist()
            present = [c for c in emission_cols if c in df.columns]
            emission = pd.to_numeric(row[present], errors="coerce").fillna(0).tolist() if present else None
            forecasts[name] = self.forecast_state(name, loss, emission)
        return forecasts

    def forecast_country(self, df: pd.DataFrame) -> dict[str, Any]:
        loss_cols = [f"tc_loss_ha_{y}" for y in PAST_YEARS]
        emission_cols = [f"gfw_gross_emissions_co2e_all_gases_{y}__Mg" for y in PAST_YEARS]
        losses = df[loss_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0).fillna(0).tolist()
        present = [c for c in emission_cols if c in df.columns]
        emissions = df[present].apply(pd.to_numeric, errors="coerce").sum(axis=0).fillna(0).tolist() if present else None
        return self.forecast_state("India", losses, emissions)


# ---------------------------------------------------------------------------
# 6. Orchestration / outputs
# ---------------------------------------------------------------------------
class ReforestationPipeline:
    # Only pre-loss structural covariates go into the regression. In particular,
    # total emissions aggregated over 2001-2020 and 2010 forest extent may be
    # temporally overlapping with the target; choose this feature list only if
    # those values are genuinely available at prediction time for your use case.
    BASE_FEATURES = [
        "area_ha",
        "extent_2000_ha",
        "forest_cover_ratio_2000",
        "biomass_density",
        "net_carbon_flux",
        "is_carbon_source",
        "log_area",
    ]

    def __init__(self):
        self.loader = DataLoader()
        self.engineer = FeatureEngineer()
        self.fuzzy = FuzzyLogicEngine()
        self.models = EnsembleModels()
        self.forecaster = TimeSeriesForecaster()
        self.state_data: pd.DataFrame | None = None
        self.results: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        print("=" * 68)
        print("REFORESTATION ML PIPELINE")
        print("=" * 68)
        self.loader.load_all()
        raw = self.loader.get_state_data()
        country_tcl, _ = self.loader.get_country_data()
        self.state_data = self.engineer.engineer_state_features(raw)

        # The priority score is a decision-support ranking, not a validated
        # supervised target. Normalize within the current set of states.
        self.state_data["_max_loss_for_priority"] = self.state_data["total_loss_ha"].max()
        self.state_data["_max_emissions_for_priority"] = self.state_data["total_emissions_Mg"].max(skipna=True)
        scores = self.state_data.apply(self.fuzzy.compute_priority, axis=1, result_type="expand")
        self.state_data = pd.concat([self.state_data, scores], axis=1)
        self.state_data.drop(
            columns=["_max_loss_for_priority", "_max_emissions_for_priority"],
            inplace=True, errors="ignore"
        )

        feature_cols = [c for c in self.BASE_FEATURES if c in self.state_data.columns]
        # Remove constant/all-null features.
        feature_cols = [
            c for c in feature_cols
            if self.state_data[c].notna().any()
            and self.state_data[c].nunique(dropna=True) > 1
        ]
        print(f"Independent regression features ({len(feature_cols)}): {feature_cols}")
        _, metrics = self.models.train_all(self.state_data, feature_cols)

        forecasts = self.forecaster.forecast_all_states(self.state_data)
        country_forecast = self.forecaster.forecast_country(self.state_data)
        loss_cols = [f"tc_loss_ha_{y}" for y in PAST_YEARS]
        emission_cols = [f"gfw_gross_emissions_co2e_all_gases_{y}__Mg" for y in PAST_YEARS]
        emissions_present = [c for c in emission_cols if c in self.state_data.columns]

        rankings = []
        for _, row in self.state_data.iterrows():
            rankings.append({
                "name": str(row["subnational1"]),
                "area_ha": float(row["area_ha"]) if pd.notna(row.get("area_ha")) else None,
                "extent_2000_ha": float(row["extent_2000_ha"]) if pd.notna(row.get("extent_2000_ha")) else None,
                "total_loss_ha": float(row["total_loss_ha"]) if pd.notna(row.get("total_loss_ha")) else None,
                "avg_annual_loss_ha": float(row["avg_annual_loss_ha"]) if pd.notna(row.get("avg_annual_loss_ha")) else None,
                "remaining_forest_ratio": float(row["remaining_forest_ratio"]) if pd.notna(row.get("remaining_forest_ratio")) else None,
                "vulnerability_score": float(row["vulnerability_score"]),
                "fuzzy_priority_score": float(row["fuzzy_priority_score"]),
                "fuzzy_priority_label": str(row["fuzzy_priority_label"]),
                "deforestation_acceleration": float(row["deforestation_acceleration"]) if pd.notna(row.get("deforestation_acceleration")) else None,
                "total_emissions_Mg": float(row["total_emissions_Mg"]) if pd.notna(row.get("total_emissions_Mg")) else None,
                "annual_losses": pd.to_numeric(row[loss_cols], errors="coerce").fillna(0).tolist(),
                "annual_emissions": pd.to_numeric(row[emissions_present], errors="coerce").fillna(0).tolist() if emissions_present else [],
            })
        rankings.sort(key=lambda item: item["vulnerability_score"], reverse=True)

        plan = []
        for state in rankings:
            f = forecasts.get(state["name"], {})
            plan.append({
                "name": state["name"],
                "priority_label": state["fuzzy_priority_label"],
                "priority_score": state["fuzzy_priority_score"],
                "projected_loss_avoidance_ha": f.get("projected_loss_avoidance_ha", 0),
                "carbon_savings_Mg": f.get("carbon_savings_Mg", 0),
                "current_loss_rate": state["avg_annual_loss_ha"],
            })
        plan.sort(key=lambda x: x["priority_score"], reverse=True)

        annual_national_loss = [
            float(pd.to_numeric(self.state_data[f"tc_loss_ha_{y}"], errors="coerce").sum())
            for y in PAST_YEARS
        ]
        annual_national_emissions = [
            float(pd.to_numeric(self.state_data[c], errors="coerce").sum())
            if c in self.state_data else 0.0
            for c in emission_cols
        ]
        total_area = float(pd.to_numeric(self.state_data["area_ha"], errors="coerce").sum())
        total_forest = float(pd.to_numeric(self.state_data["extent_2000_ha"], errors="coerce").sum())
        total_loss = float(pd.to_numeric(self.state_data["total_loss_ha"], errors="coerce").sum())
        total_gain = float(pd.to_numeric(self.state_data.get("gain_2000-2012_ha", 0), errors="coerce").sum())
        total_emissions = float(self.state_data[emissions_present].apply(pd.to_numeric, errors="coerce").sum().sum()) if emissions_present else 0.0

        overview = {
            "total_area_ha": total_area,
            "total_forest_2000_ha": total_forest,
            "total_loss_ha": total_loss,
            "total_gain_ha": total_gain,
            "net_loss_ha": total_loss - total_gain,
            "total_emissions_Mg": total_emissions,
            "forest_cover_pct_2000": round(total_forest / total_area * 100, 2) if total_area > 0 else 0,
            "avg_annual_loss_ha": total_loss / len(PAST_YEARS),
            "num_states": len(self.state_data),
            "num_critical_states": sum(x["fuzzy_priority_label"] == "Critical" for x in rankings),
            "num_high_states": sum(x["fuzzy_priority_label"] == "High" for x in rankings),
            "best_regression_model_by_oof_mae": self.models.best_model_name,
        }

        feature_importance = {}
        for model_name, importance in self.models.feature_importance.items():
            feature_importance[model_name] = [
                {"feature": k, "importance": v}
                for k, v in sorted(importance.items(), key=lambda pair: pair[1], reverse=True)
            ]

        self.results = {
            "overview": overview,
            "national_timeline": {
                "years": PAST_YEARS,
                "annual_loss_ha": annual_national_loss,
                "annual_emissions_Mg": annual_national_emissions,
                "cumulative_loss_ha": np.cumsum(annual_national_loss).tolist(),
            },
            "state_rankings": rankings,
            "state_forecasts": forecasts,
            "country_forecast": country_forecast,
            "model_performance": metrics,
            "feature_importance": feature_importance,
            "reforestation_plan": plan,
            "methodology_notes": [
                "Regression predictors exclude features directly derived from the 2001–2020 loss target.",
                "Reported regression metrics are out-of-fold estimates from shuffled K-fold cross-validation.",
                "With few state-level samples, performance estimates have high uncertainty.",
                "The reforestation forecast is an illustrative scenario, not a causal policy-impact estimate.",
                "Fuzzy/vulnerability scores are decision-support rankings, not ground-truth labels.",
            ],
        }
        self.save_results()
        print("\nPipeline complete.")
        print(f"Outputs: {OUTPUT_DIR.resolve()}")
        print(f"Models:  {MODEL_DIR.resolve()}")
        print(f"Selected model by OOF MAE: {self.models.best_model_name}")
        for name, values in metrics.items():
            if name != "_selection":
                print(f"  {name}: R²={values['r2']:.3f}, MAE={values['mae']:,.1f}, RMSE={values['rmse']:,.1f}")
        return self.results

    def save_results(self) -> None:
        write_json(OUTPUT_DIR / "pipeline_results.json", self.results)
        for key, value in self.results.items():
            write_json(OUTPUT_DIR / f"{key}.json", value)
        if self.state_data is not None:
            self.state_data.to_csv(OUTPUT_DIR / "state_features_and_scores.csv", index=False)
        for name, model in self.models.models.items():
            safe = name.lower().replace(" ", "_")
            joblib.dump(model, MODEL_DIR / f"{safe}.joblib")
        print(f"Saved outputs and {len(self.models.models)} models.")


if __name__ == "__main__":
    pipeline = ReforestationPipeline()
    result = pipeline.run()
    print("\nSUMMARY")
    print(json.dumps(result["overview"], indent=2))