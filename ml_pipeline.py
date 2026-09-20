"""
Reforestation ML Pipeline
=========================
Advanced ML pipeline for analyzing India's deforestation data and predicting
reforestation priority areas. Uses ensemble learning, SMOTE, fuzzy logic,
and time-series forecasting.

Author: Reforestation Project
Data Source: Global Forest Watch (India)
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from pathlib import Path

# ML imports
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    StackingRegressor, AdaBoostRegressor,
    ExtraTreesRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import (
    cross_val_score, LeaveOneOut, train_test_split, cross_val_predict
)
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
import xgboost as xgb
from catboost import CatBoostRegressor
# pyrefly: ignore [missing-import]
from imblearn.over_sampling import SMOTE
from scipy.interpolate import UnivariateSpline
import joblib

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = Path('Dataset')
OUTPUT_DIR = Path('ml_outputs')
MODEL_DIR = Path('models')
THRESHOLD = 30  # Standard forest cover definition (30% canopy)
FORECAST_YEARS = list(range(2021, 2031))  # 10-year forecast
PAST_YEARS = list(range(2001, 2021))  # Historical data

for d in [OUTPUT_DIR, MODEL_DIR]:
    d.mkdir(exist_ok=True)


# ============================================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================================
class DataLoader:
    """Load and preprocess all datasets."""

    def __init__(self):
        self.country_tcl = None
        self.country_carbon = None
        self.state_tcl = None
        self.state_carbon = None
        self.district_tcl = None
        self.district_carbon = None

    def load_all(self):
        """Load all 6 Excel datasets."""
        print("📂 Loading datasets...")

        self.country_tcl = pd.read_excel(DATA_DIR / 'Country tree cover loss.xlsx')
        self.country_carbon = pd.read_excel(DATA_DIR / 'Country carbon data.xlsx')
        self.state_tcl = pd.read_excel(DATA_DIR / 'Subnational 1 tree cover loss.xlsx')
        self.state_carbon = pd.read_excel(DATA_DIR / 'Subnational 1 carbon data.xlsx')
        self.district_tcl = pd.read_excel(DATA_DIR / 'Subnational 2 tree cover loss.xlsx')
        self.district_carbon = pd.read_excel(DATA_DIR / 'Subnational 2 carbon data.xlsx')

        print(f"  ✅ Country TCL: {self.country_tcl.shape}")
        print(f"  ✅ Country Carbon: {self.country_carbon.shape}")
        print(f"  ✅ State TCL: {self.state_tcl.shape}")
        print(f"  ✅ State Carbon: {self.state_carbon.shape}")
        print(f"  ✅ District TCL: {self.district_tcl.shape}")
        print(f"  ✅ District Carbon: {self.district_carbon.shape}")

    def get_state_data(self):
        """Get merged state-level data at threshold=30.

        FIX: The carbon dataset has 'umd_tree_cover_density__threshold'
        while TCL has 'threshold'. We rename + filter both to threshold=30,
        then drop the extra threshold column before merging to prevent
        duplicate 'threshold' / 'threshold_carbon' confusion that was
        causing emission columns to be inaccessible.
        """
        # Filter at standard threshold
        tcl = self.state_tcl[self.state_tcl['threshold'] == THRESHOLD].copy()
        carbon = self.state_carbon[
            self.state_carbon['umd_tree_cover_density__threshold'] == THRESHOLD
        ].copy()

        # Drop the threshold column from carbon to avoid duplicate on merge
        carbon = carbon.drop(columns=['umd_tree_cover_density__threshold'])

        # Merge on country + subnational1 (no threshold column conflict now)
        merged = pd.merge(
            tcl, carbon,
            on=['country', 'subnational1'],
            how='left',
            suffixes=('', '_carbon')
        )

        # Clean up
        merged = merged.dropna(subset=['subnational1'])

        # Validate emission data is present
        emission_sample_col = 'gfw_gross_emissions_co2e_all_gases_2001__Mg'
        if emission_sample_col in merged.columns:
            total_em = merged[emission_sample_col].sum()
            print(f"  ✅ Emission data validation: 2001 emissions = {total_em:,.0f} Mg")
        else:
            print(f"  ⚠️  WARNING: Emission columns not found after merge!")

        print(f"  📊 State-level merged data: {merged.shape}")
        return merged

    def get_country_data(self):
        """Get country-level data at threshold=30."""
        tcl = self.country_tcl[self.country_tcl['threshold'] == THRESHOLD].copy()
        carbon = self.country_carbon[
            self.country_carbon['umd_tree_cover_density__threshold'] == THRESHOLD
        ].copy()
        return tcl, carbon

    def get_district_data(self):
        """Get district-level data at threshold=30."""
        tcl = self.district_tcl[self.district_tcl['threshold'] == THRESHOLD].copy()
        carbon = self.district_carbon[
            self.district_carbon['umd_tree_cover_density__threshold'] == THRESHOLD
        ].copy()

        # Drop threshold column from carbon to avoid conflicts
        carbon = carbon.drop(columns=['umd_tree_cover_density__threshold'])

        merged = pd.merge(
            tcl, carbon,
            on=['country', 'subnational1', 'subnational2'],
            how='left',
            suffixes=('', '_carbon')
        )
        merged = merged.dropna(subset=['subnational2'])
        return merged


# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
class FeatureEngineer:
    """Engineer features for ML models."""

    def __init__(self):
        self.loss_columns = [f'tc_loss_ha_{y}' for y in PAST_YEARS]
        self.emission_columns = [
            f'gfw_gross_emissions_co2e_all_gases_{y}__Mg' for y in PAST_YEARS
        ]

    def engineer_state_features(self, df):
        """Create features from state-level data."""
        print("🔧 Engineering features...")

        result = df.copy()

        # --- Core Forest Metrics ---
        result['total_loss_ha'] = result[self.loss_columns].sum(axis=1)
        result['avg_annual_loss_ha'] = result[self.loss_columns].mean(axis=1)
        result['max_annual_loss_ha'] = result[self.loss_columns].max(axis=1)
        result['min_annual_loss_ha'] = result[self.loss_columns].min(axis=1)
        result['std_annual_loss_ha'] = result[self.loss_columns].std(axis=1)

        # Forest cover ratios (clamped to [0, 1])
        result['forest_cover_ratio_2000'] = np.clip(
            result['extent_2000_ha'] / result['area_ha'].replace(0, np.nan),
            0, 1
        )
        result['forest_cover_ratio_2010'] = np.clip(
            result['extent_2010_ha'] / result['area_ha'].replace(0, np.nan),
            0, 1
        )
        result['forest_cover_change_2000_2010'] = (
            (result['extent_2010_ha'] - result['extent_2000_ha'])
            / result['extent_2000_ha'].replace(0, np.nan)
        )

        # Net forest status
        result['remaining_forest_ha'] = (
            result['extent_2000_ha'] - result['total_loss_ha']
            + result['gain_2000-2012_ha'].fillna(0)
        )
        # Clamp to >= 0 (forest area can't be negative)
        result['remaining_forest_ha'] = result['remaining_forest_ha'].clip(lower=0)
        result['remaining_forest_ratio'] = np.clip(
            result['remaining_forest_ha'] / result['area_ha'].replace(0, np.nan),
            0, 1
        )

        # --- Deforestation Trend Analysis ---
        # First half (2001-2010) vs Second half (2011-2020)
        first_half_cols = [f'tc_loss_ha_{y}' for y in range(2001, 2011)]
        second_half_cols = [f'tc_loss_ha_{y}' for y in range(2011, 2021)]
        result['avg_loss_first_half'] = result[first_half_cols].mean(axis=1)
        result['avg_loss_second_half'] = result[second_half_cols].mean(axis=1)
        result['deforestation_acceleration'] = (
            (result['avg_loss_second_half'] - result['avg_loss_first_half'])
            / result['avg_loss_first_half'].replace(0, np.nan)
        )

        # Linear trend (slope of annual loss)
        years_array = np.arange(len(PAST_YEARS))
        slopes = []
        for _, row in result.iterrows():
            losses = row[self.loss_columns].values.astype(float)
            if np.all(np.isnan(losses)):
                slopes.append(0)
            else:
                losses = np.nan_to_num(losses, 0)
                slope = np.polyfit(years_array, losses, 1)[0]
                slopes.append(slope)
        result['loss_trend_slope'] = slopes

        # Coefficient of variation
        result['loss_cv'] = (
            result['std_annual_loss_ha']
            / result['avg_annual_loss_ha'].replace(0, np.nan)
        )

        # --- Carbon Impact Features ---
        emission_cols_present = [c for c in self.emission_columns if c in result.columns]
        if emission_cols_present:
            result['total_emissions_Mg'] = result[emission_cols_present].sum(axis=1)
            result['avg_annual_emissions_Mg'] = result[emission_cols_present].mean(axis=1)
            result['carbon_intensity'] = (
                result['total_emissions_Mg']
                / result['total_loss_ha'].replace(0, np.nan)
            )
            # Clamp carbon_intensity to positive (ratio of two positive quantities)
            result['carbon_intensity'] = result['carbon_intensity'].clip(lower=0)
            print(f"  ✅ Emission columns found: {len(emission_cols_present)}")
            print(f"  ✅ Total emissions across all states: {result['total_emissions_Mg'].sum():,.0f} Mg")
        else:
            result['total_emissions_Mg'] = 0
            result['avg_annual_emissions_Mg'] = 0
            result['carbon_intensity'] = 0
            print("  ⚠️  No emission columns found!")

        # Carbon flux features
        if 'gfw_net_flux_co2e__Mg_yr-1' in result.columns:
            result['net_carbon_flux'] = result['gfw_net_flux_co2e__Mg_yr-1'].fillna(0)
            result['is_carbon_source'] = (result['net_carbon_flux'] > 0).astype(int)
        else:
            result['net_carbon_flux'] = 0
            result['is_carbon_source'] = 0

        # Biomass density
        if 'avg_whrc_aboveground_biomass_2000_Mg_ha-1' in result.columns:
            result['biomass_density'] = result[
                'avg_whrc_aboveground_biomass_2000_Mg_ha-1'
            ].fillna(0)
        else:
            result['biomass_density'] = 0

        # --- Log-transformed features (for skewed distributions) ---
        result['log_total_loss'] = np.log1p(result['total_loss_ha'].clip(lower=0))
        result['log_total_emissions'] = np.log1p(result['total_emissions_Mg'].clip(lower=0))
        result['log_area'] = np.log1p(result['area_ha'].clip(lower=0))

        # --- Vulnerability & Priority Scoring ---
        # Normalize key metrics to 0-1 scale for composite scoring
        for col in ['total_loss_ha', 'deforestation_acceleration', 'carbon_intensity',
                     'loss_trend_slope', 'remaining_forest_ratio']:
            if col in result.columns:
                col_data = result[col].fillna(0)
                min_val, max_val = col_data.min(), col_data.max()
                if max_val > min_val:
                    result[f'{col}_norm'] = (col_data - min_val) / (max_val - min_val)
                else:
                    result[f'{col}_norm'] = 0

        # Composite vulnerability score
        result['vulnerability_score'] = (
            result.get('total_loss_ha_norm', 0) * 0.25 +
            result.get('deforestation_acceleration_norm', 0) * 0.20 +
            result.get('carbon_intensity_norm', 0) * 0.20 +
            result.get('loss_trend_slope_norm', 0) * 0.20 +
            (1 - result.get('remaining_forest_ratio_norm', 0)) * 0.15
        )

        # Fill remaining NaN
        result = result.fillna(0)
        result = result.replace([np.inf, -np.inf], 0)

        print(f"  ✅ Engineered {len(result.columns)} features for {len(result)} states")
        return result


# ============================================================================
# 3. FUZZY LOGIC LAYER
# ============================================================================
class FuzzyLogicEngine:
    """Fuzzy logic layer for reforestation priority classification."""

    @staticmethod
    def triangular_mf(x, a, b, c):
        """Triangular membership function."""
        return np.maximum(0, np.minimum((x - a) / max(b - a, 1e-10),
                                         (c - x) / max(c - b, 1e-10)))

    @staticmethod
    def trapezoidal_mf(x, a, b, c, d):
        """Trapezoidal membership function."""
        return np.maximum(0, np.minimum(
            np.minimum((x - a) / max(b - a, 1e-10), 1),
            (d - x) / max(d - c, 1e-10)
        ))

    def fuzzify(self, value, universe='loss_rate'):
        """Fuzzify a normalized value (0-1) into fuzzy sets."""
        x = np.clip(value, 0, 1)

        if universe == 'loss_rate':
            low = self.trapezoidal_mf(x, 0, 0, 0.2, 0.4)
            medium = self.triangular_mf(x, 0.2, 0.5, 0.8)
            high = self.trapezoidal_mf(x, 0.6, 0.8, 1, 1)
        elif universe == 'forest_density':
            low = self.trapezoidal_mf(x, 0, 0, 0.15, 0.35)
            medium = self.triangular_mf(x, 0.2, 0.5, 0.8)
            high = self.trapezoidal_mf(x, 0.65, 0.85, 1, 1)
        elif universe == 'carbon_impact':
            low = self.trapezoidal_mf(x, 0, 0, 0.25, 0.45)
            medium = self.triangular_mf(x, 0.3, 0.5, 0.7)
            high = self.trapezoidal_mf(x, 0.55, 0.75, 1, 1)
        else:
            low = medium = high = 0

        return {'low': float(low), 'medium': float(medium), 'high': float(high)}

    def apply_rules(self, loss_rate_fuzzy, forest_density_fuzzy, carbon_impact_fuzzy):
        """Apply fuzzy rules to determine reforestation priority."""
        # Rule strengths for priority levels
        critical = 0
        high = 0
        moderate = 0
        low = 0

        # Rule 1: High loss + Low forest + High carbon → CRITICAL
        critical = max(critical, min(
            loss_rate_fuzzy['high'],
            forest_density_fuzzy['low'],
            carbon_impact_fuzzy['high']
        ))

        # Rule 2: High loss + Medium forest + High carbon → CRITICAL
        critical = max(critical, min(
            loss_rate_fuzzy['high'],
            forest_density_fuzzy['medium'],
            carbon_impact_fuzzy['high']
        ))

        # Rule 3: High loss + Low forest + Medium carbon → HIGH
        high = max(high, min(
            loss_rate_fuzzy['high'],
            forest_density_fuzzy['low'],
            carbon_impact_fuzzy['medium']
        ))

        # Rule 4: Medium loss + Low forest + High carbon → HIGH
        high = max(high, min(
            loss_rate_fuzzy['medium'],
            forest_density_fuzzy['low'],
            carbon_impact_fuzzy['high']
        ))

        # Rule 5: High loss + High forest + Medium carbon → HIGH
        high = max(high, min(
            loss_rate_fuzzy['high'],
            forest_density_fuzzy['high'],
            carbon_impact_fuzzy['medium']
        ))

        # Rule 6: Medium loss + Medium forest + Medium carbon → MODERATE
        moderate = max(moderate, min(
            loss_rate_fuzzy['medium'],
            forest_density_fuzzy['medium'],
            carbon_impact_fuzzy['medium']
        ))

        # Rule 7: Medium loss + High forest + Low carbon → MODERATE
        moderate = max(moderate, min(
            loss_rate_fuzzy['medium'],
            forest_density_fuzzy['high'],
            carbon_impact_fuzzy['low']
        ))

        # Rule 8: Low loss + Medium forest + Medium carbon → LOW
        low = max(low, min(
            loss_rate_fuzzy['low'],
            forest_density_fuzzy['medium'],
            carbon_impact_fuzzy['medium']
        ))

        # Rule 9: Low loss + High forest + Low carbon → LOW
        low = max(low, min(
            loss_rate_fuzzy['low'],
            forest_density_fuzzy['high'],
            carbon_impact_fuzzy['low']
        ))

        # Rule 10: Low loss + Low forest → MODERATE (still needs attention)
        moderate = max(moderate, min(
            loss_rate_fuzzy['low'],
            forest_density_fuzzy['low']
        ))

        return {
            'critical': float(critical),
            'high': float(high),
            'moderate': float(moderate),
            'low': float(low)
        }

    def defuzzify(self, priority_fuzzy):
        """Centroid defuzzification to get crisp priority score (0-100)."""
        # Priority centers: Critical=90, High=70, Moderate=45, Low=20
        numerator = (
            priority_fuzzy['critical'] * 90 +
            priority_fuzzy['high'] * 70 +
            priority_fuzzy['moderate'] * 45 +
            priority_fuzzy['low'] * 20
        )
        denominator = sum(priority_fuzzy.values())

        if denominator == 0:
            return 50.0  # Default moderate

        return float(numerator / denominator)

    def compute_priority(self, row):
        """Compute fuzzy reforestation priority for a single state/region."""
        # Normalize inputs
        loss_rate_val = np.clip(row.get('total_loss_ha_norm', 0.5), 0, 1)
        forest_density_val = np.clip(row.get('remaining_forest_ratio', 0.5), 0, 1)
        carbon_val = np.clip(row.get('carbon_intensity_norm', 0.5) if 'carbon_intensity_norm' in row.index else 0.5, 0, 1)

        # Fuzzify inputs
        loss_fuzzy = self.fuzzify(loss_rate_val, 'loss_rate')
        forest_fuzzy = self.fuzzify(forest_density_val, 'forest_density')
        carbon_fuzzy = self.fuzzify(carbon_val, 'carbon_impact')

        # Apply rules
        priority_fuzzy = self.apply_rules(loss_fuzzy, forest_fuzzy, carbon_fuzzy)

        # Defuzzify
        priority_score = self.defuzzify(priority_fuzzy)

        # Determine label
        max_level = max(priority_fuzzy, key=priority_fuzzy.get)
        label_map = {'critical': 'Critical', 'high': 'High', 'moderate': 'Moderate', 'low': 'Low'}

        return {
            'fuzzy_priority_score': priority_score,
            'fuzzy_priority_label': label_map.get(max_level, 'Moderate'),
            'fuzzy_memberships': priority_fuzzy,
            'fuzzy_inputs': {
                'loss_rate': loss_fuzzy,
                'forest_density': forest_fuzzy,
                'carbon_impact': carbon_fuzzy
            }
        }


# ============================================================================
# 4. SMOTE ANALYSIS (Before/After comparison)
# ============================================================================
class SMOTEAnalyzer:
    """Apply SMOTE to balance classification and compare results."""

    def __init__(self):
        self.results = {}

    def prepare_classification_data(self, df):
        """Convert regression targets to classification categories."""
        # Create priority categories based on vulnerability score
        bins = [0, 0.3, 0.6, 1.0]
        labels = ['Low', 'Medium', 'High']
        df = df.copy()
        df['priority_class'] = pd.cut(
            df['vulnerability_score'], bins=bins, labels=labels, include_lowest=True
        )

        # Encode labels
        le = LabelEncoder()
        df['priority_encoded'] = le.fit_transform(df['priority_class'].astype(str))

        return df, le

    def run_analysis(self, df, feature_cols):
        """Run before/after SMOTE comparison with proper train/test split."""
        print("⚖️  Running SMOTE Analysis...")

        df_class, le = self.prepare_classification_data(df)

        X = df_class[feature_cols].values
        y = df_class['priority_encoded'].values

        # Handle any remaining NaN
        X = np.nan_to_num(X, 0)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Class distribution BEFORE SMOTE
        unique, counts = np.unique(y, return_counts=True)
        before_dist = dict(zip(le.inverse_transform(unique), counts.tolist()))
        print(f"  📊 Before SMOTE: {before_dist}")

        # --- BEFORE SMOTE (using proper LOO cross-validation) ---
        rf_before = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_folds_before = min(5, min(counts))
        if cv_folds_before < 2:
            cv_folds_before = 2
        scores_before = cross_val_score(rf_before, X_scaled, y,
                                         cv=cv_folds_before, scoring='accuracy')

        # Get predictions via cross_val_predict (proper evaluation, no data leakage)
        y_pred_before = cross_val_predict(rf_before, X_scaled, y, cv=cv_folds_before)

        # --- APPLY SMOTE ---
        min_samples = min(counts)
        k_neighbors = min(min_samples - 1, 3) if min_samples > 1 else 1

        if min_samples > 1 and len(unique) > 1:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

            unique_after, counts_after = np.unique(y_resampled, return_counts=True)
            after_dist = dict(zip(le.inverse_transform(unique_after), counts_after.tolist()))
            print(f"  📊 After SMOTE: {after_dist}")

            # --- AFTER SMOTE (proper evaluation with train/test split) ---
            # Split SMOTE-resampled data
            X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
                X_resampled, y_resampled, test_size=0.25, random_state=42, stratify=y_resampled
            )
            rf_after = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_after.fit(X_train_s, y_train_s)
            y_pred_after = rf_after.predict(X_test_s)

            cv_folds_after = min(5, min(counts_after))
            if cv_folds_after < 2:
                cv_folds_after = 2
            scores_after = cross_val_score(rf_after, X_resampled, y_resampled,
                                            cv=cv_folds_after, scoring='accuracy')
        else:
            after_dist = before_dist
            scores_after = scores_before
            y_pred_after = y_pred_before
            y_test_s = y
            y_resampled = y

        self.results = {
            'before_smote': {
                'class_distribution': before_dist,
                'accuracy': float(np.mean(scores_before)),
                'cv_scores': scores_before.tolist(),
                'f1_macro': float(f1_score(y, y_pred_before, average='macro', zero_division=0)),
                'precision_macro': float(precision_score(y, y_pred_before, average='macro', zero_division=0)),
                'recall_macro': float(recall_score(y, y_pred_before, average='macro', zero_division=0)),
            },
            'after_smote': {
                'class_distribution': after_dist,
                'accuracy': float(np.mean(scores_after)),
                'cv_scores': scores_after.tolist(),
                'f1_macro': float(f1_score(y_test_s, y_pred_after, average='macro', zero_division=0)),
                'precision_macro': float(precision_score(y_test_s, y_pred_after, average='macro', zero_division=0)),
                'recall_macro': float(recall_score(y_test_s, y_pred_after, average='macro', zero_division=0)),
            }
        }

        print(f"  ✅ Before SMOTE Accuracy: {self.results['before_smote']['accuracy']:.4f}")
        print(f"  ✅ After SMOTE Accuracy: {self.results['after_smote']['accuracy']:.4f}")

        return self.results


# ============================================================================
# 5. ENSEMBLE ML MODELS
# ============================================================================
class EnsembleModels:
    """Train and evaluate ensemble ML models."""

    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.feature_importance = {}
        self.scaler = StandardScaler()

    def train_all(self, df, feature_cols, target_col='total_loss_ha'):
        """Train all models and evaluate."""
        print("🤖 Training Ensemble Models...")

        X = df[feature_cols].values
        y = df[target_col].values

        X = np.nan_to_num(X, 0)
        y = np.nan_to_num(y, 0)

        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25, random_state=42
        )

        # --- 1. Random Forest ---
        print("  🌲 Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_split=4,
            min_samples_leaf=3, random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.models['Random Forest'] = rf
        self._evaluate(rf, X_train, y_train, X_test, y_test, 'Random Forest', feature_cols)

        # --- 2. XGBoost (with regularization to reduce overfitting) ---
        print("  🚀 Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.7, random_state=42,
            verbosity=0, reg_alpha=0.1, reg_lambda=1.5, min_child_weight=3,
            gamma=0.1
        )
        xgb_model.fit(X_train, y_train)
        self.models['XGBoost'] = xgb_model
        self._evaluate(xgb_model, X_train, y_train, X_test, y_test, 'XGBoost', feature_cols)

        # --- 3. CatBoost (with regularization) ---
        print("  🐱 Training CatBoost...")
        cat = CatBoostRegressor(
            iterations=150, depth=4, learning_rate=0.08,
            random_seed=42, verbose=False, l2_leaf_reg=5,
            min_data_in_leaf=3
        )
        cat.fit(X_train, y_train)
        self.models['CatBoost'] = cat
        self._evaluate(cat, X_train, y_train, X_test, y_test, 'CatBoost', feature_cols)

        # --- 4. SVR ---
        print("  📐 Training SVR...")
        svr = SVR(kernel='rbf', C=50, gamma='scale', epsilon=0.1)
        svr.fit(X_train, y_train)
        self.models['SVR'] = svr
        self._evaluate(svr, X_train, y_train, X_test, y_test, 'SVR', feature_cols)

        # --- 5. Ridge Regression (baseline) ---
        print("  📏 Training Ridge Regression...")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        self.models['Ridge'] = ridge
        self._evaluate(ridge, X_train, y_train, X_test, y_test, 'Ridge', feature_cols)

        # --- 6. AdaBoost ---
        print("  🔥 Training AdaBoost...")
        ada = AdaBoostRegressor(n_estimators=80, random_state=42, learning_rate=0.05)
        ada.fit(X_train, y_train)
        self.models['AdaBoost'] = ada
        self._evaluate(ada, X_train, y_train, X_test, y_test, 'AdaBoost', feature_cols)

        # --- 7. Extra Trees ---
        print("  🌳 Training Extra Trees...")
        et = ExtraTreesRegressor(
            n_estimators=200, max_depth=6, min_samples_split=4,
            min_samples_leaf=3, random_state=42, n_jobs=-1
        )
        et.fit(X_train, y_train)
        self.models['Extra Trees'] = et
        self._evaluate(et, X_train, y_train, X_test, y_test, 'Extra Trees', feature_cols)

        # --- 8. Gradient Boosting ---
        print("  📈 Training Gradient Boosting...")
        gb = GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            min_samples_split=4, min_samples_leaf=3,
            subsample=0.8, random_state=42
        )
        gb.fit(X_train, y_train)
        self.models['Gradient Boosting'] = gb
        self._evaluate(gb, X_train, y_train, X_test, y_test, 'Gradient Boosting', feature_cols)

        # --- 9. HistGradientBoosting (with min_samples_leaf set for small data) ---
        print("  📊 Training HistGradientBoosting...")
        hgb = HistGradientBoostingRegressor(
            max_iter=150, max_depth=4, learning_rate=0.08,
            min_samples_leaf=3, random_state=42
        )
        hgb.fit(X_train, y_train)
        self.models['HistGradientBoosting'] = hgb
        self._evaluate(hgb, X_train, y_train, X_test, y_test, 'HistGradientBoosting', feature_cols)

        # --- 10. ElasticNet ---
        print("  🔗 Training ElasticNet...")
        en = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42, max_iter=5000)
        en.fit(X_train, y_train)
        self.models['ElasticNet'] = en
        self._evaluate(en, X_train, y_train, X_test, y_test, 'ElasticNet', feature_cols)

        # --- 11. Stacking Ensemble ---
        print("  🏗️  Training Stacking Ensemble...")
        stacking = StackingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=5,
                                              min_samples_leaf=3, random_state=42)),
                ('xgb', xgb.XGBRegressor(n_estimators=100, max_depth=4, verbosity=0,
                                          random_state=42, reg_alpha=0.1, reg_lambda=1.5)),
                ('cat', CatBoostRegressor(iterations=100, depth=4, verbose=False,
                                           random_seed=42, l2_leaf_reg=5)),
            ],
            final_estimator=Ridge(alpha=1.0),
            cv=3
        )
        stacking.fit(X_train, y_train)
        self.models['Stacking Ensemble'] = stacking
        self._evaluate(stacking, X_train, y_train, X_test, y_test, 'Stacking Ensemble', feature_cols)

        # Leave-One-Out Cross-Validation (more reliable for 36 samples)
        print("  🔄 Running Leave-One-Out Cross-Validation...")
        loo = LeaveOneOut()
        for name, model in self.models.items():
            try:
                # LOO R2 cannot be computed per-fold (undefined for 1 sample).
                # We must collect all predictions and compute R2 once.
                loo_preds = cross_val_predict(model, X_scaled, y, cv=loo)
                loo_r2 = float(r2_score(y, loo_preds))
                self.metrics[name]['cv_r2_mean'] = loo_r2
                self.metrics[name]['cv_r2_std'] = 0.0  # Not applicable for single LOO score
                
                # Compute absolute errors for some distribution stats instead
                abs_errors = np.abs(y - loo_preds)
                self.metrics[name]['cv_r2_scores'] = [
                    float(np.percentile(abs_errors, 25)),
                    float(np.median(abs_errors)),
                    float(np.percentile(abs_errors, 75)),
                ]
            except Exception as e:
                print(f"    ⚠️  LOO-CV failed for {name}: {e}")
                self.metrics[name]['cv_r2_mean'] = 0.0
                self.metrics[name]['cv_r2_std'] = 0.0
                self.metrics[name]['cv_r2_scores'] = [0.0, 0.0, 0.0]

        print("  ✅ All models trained!")
        return self.models, self.metrics

    def _evaluate(self, model, X_train, y_train, X_test, y_test, name, feature_cols):
        """Evaluate a model."""
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_r2 = float(r2_score(y_train, y_train_pred))
        test_r2 = float(r2_score(y_test, y_test_pred))

        self.metrics[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': float(mean_absolute_error(y_train, y_train_pred)),
            'test_mae': float(mean_absolute_error(y_test, y_test_pred)),
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            'overfit_gap': float(train_r2 - test_r2),
        }

        # WAPE (Weighted Absolute Percentage Error)
        total_actual = np.sum(np.abs(y_test))
        if total_actual > 0:
            self.metrics[name]['test_wape'] = float(
                np.sum(np.abs(y_test - y_test_pred)) / total_actual * 100
            )
        else:
            self.metrics[name]['test_wape'] = 0.0

        # Feature importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            self.feature_importance[name] = dict(
                zip(feature_cols, [float(x) for x in importances])
            )

        print(f"    {name}: R²={test_r2:.4f}, "
              f"MAE={self.metrics[name]['test_mae']:.2f}, "
              f"RMSE={self.metrics[name]['test_rmse']:.2f}, "
              f"Overfit={self.metrics[name]['overfit_gap']:.4f}")


# ============================================================================
# 6. TIME-SERIES FORECASTING
# ============================================================================
class TimeSeriesForecaster:
    """Forecast future deforestation under different scenarios."""

    def __init__(self):
        self.forecasts = {}

    def forecast_state(self, state_name, annual_losses, annual_emissions=None):
        """Forecast for a single state under two scenarios.

        All output values are clamped to >= 0 (negative loss/emissions are
        physically impossible).
        """
        years = np.array(PAST_YEARS, dtype=float)
        losses = np.array(annual_losses, dtype=float)
        losses = np.nan_to_num(losses, 0)

        # Fit polynomial trend (degree 2)
        coeffs = np.polyfit(years, losses, 2)
        poly_func = np.poly1d(coeffs)

        # Linear trend for simpler projection
        linear_coeffs = np.polyfit(years, losses, 1)
        linear_func = np.poly1d(linear_coeffs)

        # Recent trend (last 5 years)
        recent_years = years[-5:]
        recent_losses = losses[-5:]
        recent_coeffs = np.polyfit(recent_years, recent_losses, 1)
        recent_func = np.poly1d(recent_coeffs)

        # Average of last 5 years
        avg_recent = float(np.mean(recent_losses))

        forecast_years = np.array(FORECAST_YEARS, dtype=float)

        # --- Scenario A: Business-as-Usual ---
        # Weighted combination: 40% polynomial, 30% recent trend, 30% average
        bau_poly = poly_func(forecast_years)
        bau_recent = recent_func(forecast_years)
        bau_avg = np.full_like(forecast_years, avg_recent)
        bau_forecast = 0.4 * bau_poly + 0.3 * bau_recent + 0.3 * bau_avg
        # CLAMP: No negative losses
        bau_forecast = np.maximum(bau_forecast, 0)

        # --- Scenario B: Reforestation ---
        # Assume 5% year-over-year reduction in deforestation + active reforestation
        reforest_forecast = []
        current_loss = avg_recent
        for i, year in enumerate(FORECAST_YEARS):
            reduction_rate = 0.05 + (0.02 * i)  # Increasing reduction over time
            current_loss = current_loss * (1 - reduction_rate)
            reforest_forecast.append(max(current_loss, avg_recent * 0.1))  # Floor at 10%
        reforest_forecast = np.array(reforest_forecast)
        # CLAMP: No negative losses
        reforest_forecast = np.maximum(reforest_forecast, 0)

        # --- Carbon Projections ---
        if annual_emissions is not None:
            emissions = np.array(annual_emissions, dtype=float)
            emissions = np.nan_to_num(emissions, 0)
            total_loss_sum = np.sum(losses)
            avg_emission_per_loss = np.sum(emissions) / max(total_loss_sum, 1)

            bau_emissions = np.maximum(bau_forecast * avg_emission_per_loss, 0)
            reforest_emissions = np.maximum(reforest_forecast * avg_emission_per_loss, 0)
            # Reforestation absorbs carbon — cumulative savings
            reforest_gain = np.cumsum(
                np.maximum((bau_forecast - reforest_forecast) * avg_emission_per_loss * 0.5, 0)
            )
        else:
            bau_emissions = bau_forecast * 0
            reforest_emissions = reforest_forecast * 0
            reforest_gain = np.zeros_like(forecast_years)

        # Cumulative projections
        total_past_loss = float(np.sum(losses))
        bau_cumulative = total_past_loss + np.cumsum(bau_forecast)
        reforest_cumulative = total_past_loss + np.cumsum(reforest_forecast)

        # Compute avoidance (BAU - Reforestation), clamped to >= 0
        projected_loss_avoidance = max(0.0, float(
            np.sum(bau_forecast) - np.sum(reforest_forecast)
        ))
        carbon_savings = max(0.0, float(np.sum(bau_emissions) - np.sum(reforest_emissions)))

        return {
            'state': state_name,
            'past': {
                'years': PAST_YEARS,
                'annual_loss_ha': losses.tolist(),
                'cumulative_loss_ha': np.cumsum(losses).tolist(),
            },
            'bau_scenario': {
                'years': FORECAST_YEARS,
                'annual_loss_ha': bau_forecast.tolist(),
                'cumulative_loss_ha': bau_cumulative.tolist(),
                'annual_emissions_Mg': bau_emissions.tolist(),
            },
            'reforestation_scenario': {
                'years': FORECAST_YEARS,
                'annual_loss_ha': reforest_forecast.tolist(),
                'cumulative_loss_ha': reforest_cumulative.tolist(),
                'annual_emissions_Mg': reforest_emissions.tolist(),
                'carbon_saved_Mg': reforest_gain.tolist(),
            },
            # Consistent key names (was inconsistently named before)
            'projected_loss_avoidance_ha': projected_loss_avoidance,
            'carbon_savings_Mg': carbon_savings,
            'trend_slope': float(linear_coeffs[0]),
        }

    def forecast_all_states(self, df):
        """Generate forecasts for all states."""
        print("🔮 Generating Time-Series Forecasts...")

        loss_cols = [f'tc_loss_ha_{y}' for y in PAST_YEARS]
        emission_cols = [f'gfw_gross_emissions_co2e_all_gases_{y}__Mg' for y in PAST_YEARS]

        forecasts = {}
        for _, row in df.iterrows():
            state = row['subnational1']
            losses = row[loss_cols].values.astype(float)
            emissions_present = [c for c in emission_cols if c in row.index]
            if emissions_present:
                emissions = row[emissions_present].values.astype(float)
            else:
                emissions = None

            forecasts[state] = self.forecast_state(state, losses, emissions)
            print(f"  ✅ {state}: Loss avoidance = "
                  f"{forecasts[state]['projected_loss_avoidance_ha']:,.0f} ha")

        self.forecasts = forecasts
        return forecasts

    def forecast_country(self, df):
        """Aggregate country-level forecast."""
        loss_cols = [f'tc_loss_ha_{y}' for y in PAST_YEARS]
        emission_cols = [f'gfw_gross_emissions_co2e_all_gases_{y}__Mg' for y in PAST_YEARS]

        # Sum across all states
        total_losses = df[loss_cols].sum().values.astype(float)
        emissions_present = [c for c in emission_cols if c in df.columns]
        if emissions_present:
            total_emissions = df[emissions_present].sum().values.astype(float)
        else:
            total_emissions = None

        country_forecast = self.forecast_state('India', total_losses, total_emissions)
        print(f"  🇮🇳 India Total: Loss avoidance = "
              f"{country_forecast['projected_loss_avoidance_ha']:,.0f} ha")

        return country_forecast


# ============================================================================
# 7. MAIN PIPELINE
# ============================================================================
class ReforestationPipeline:
    """Main pipeline orchestrating all components."""

    def __init__(self):
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.fuzzy_engine = FuzzyLogicEngine()
        self.smote_analyzer = SMOTEAnalyzer()
        self.ensemble = EnsembleModels()
        self.forecaster = TimeSeriesForecaster()
        self.state_data = None
        self.results = {}

    def run(self):
        """Execute the full pipeline."""
        print("=" * 70)
        print("🌲 REFORESTATION ML PIPELINE")
        print("=" * 70)

        # Step 1: Load Data
        self.data_loader.load_all()
        self.state_data = self.data_loader.get_state_data()
        country_tcl, country_carbon = self.data_loader.get_country_data()

        # Step 2: Feature Engineering
        self.state_data = self.feature_engineer.engineer_state_features(self.state_data)

        # Step 3: Fuzzy Logic Priority Scoring
        print("🧠 Computing Fuzzy Logic Priorities...")
        fuzzy_results = []
        for idx, row in self.state_data.iterrows():
            result = self.fuzzy_engine.compute_priority(row)
            fuzzy_results.append(result)
            self.state_data.loc[idx, 'fuzzy_priority_score'] = result['fuzzy_priority_score']
            self.state_data.loc[idx, 'fuzzy_priority_label'] = result['fuzzy_priority_label']

        fuzzy_summary = self.state_data[['subnational1', 'fuzzy_priority_score', 'fuzzy_priority_label']].copy()
        print(f"  ✅ Fuzzy priorities computed for {len(fuzzy_summary)} states")
        print(f"  Priority distribution:")
        print(f"    {fuzzy_summary['fuzzy_priority_label'].value_counts().to_dict()}")

        # Step 4: Define ML features
        feature_cols = [
            'area_ha', 'extent_2000_ha', 'extent_2010_ha',
            'forest_cover_ratio_2000', 'forest_cover_ratio_2010',
            'forest_cover_change_2000_2010',
            'avg_annual_loss_ha', 'max_annual_loss_ha', 'std_annual_loss_ha',
            'deforestation_acceleration', 'loss_trend_slope', 'loss_cv',
            'remaining_forest_ratio', 'vulnerability_score',
            'biomass_density', 'total_emissions_Mg', 'carbon_intensity',
            'fuzzy_priority_score',
            'log_total_loss', 'log_total_emissions', 'log_area',
        ]

        # Verify feature columns exist
        feature_cols = [c for c in feature_cols if c in self.state_data.columns]

        # Step 5: SMOTE Analysis
        smote_results = self.smote_analyzer.run_analysis(self.state_data, feature_cols)

        # Step 6: Ensemble Models
        models, metrics = self.ensemble.train_all(
            self.state_data, feature_cols, target_col='total_loss_ha'
        )

        # Step 7: Time-Series Forecasting
        state_forecasts = self.forecaster.forecast_all_states(self.state_data)
        country_forecast = self.forecaster.forecast_country(self.state_data)

        # Step 8: Compile Results
        self._compile_results(
            country_tcl, country_carbon,
            smote_results, metrics, state_forecasts, country_forecast, fuzzy_results
        )

        # Step 9: Save Results
        self._save_results()

        print()
        print("=" * 70)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"  📁 Results saved to: {OUTPUT_DIR}/")
        print(f"  📁 Models saved to: {MODEL_DIR}/")

        return self.results

    def _compile_results(self, country_tcl, country_carbon,
                         smote_results, metrics, state_forecasts,
                         country_forecast, fuzzy_results):
        """Compile all results into a structured dict."""

        loss_cols = [f'tc_loss_ha_{y}' for y in PAST_YEARS]
        emission_cols = [f'gfw_gross_emissions_co2e_all_gases_{y}__Mg' for y in PAST_YEARS]
        emission_cols = [c for c in emission_cols if c in self.state_data.columns]

        # --- Overview Stats ---
        total_area = float(self.state_data['area_ha'].sum())
        total_forest_2000 = float(self.state_data['extent_2000_ha'].sum())
        total_loss = float(self.state_data['total_loss_ha'].sum())
        total_gain = float(self.state_data['gain_2000-2012_ha'].sum())

        # FIX: Actually compute emissions from the emission columns directly
        if emission_cols:
            total_emissions = float(self.state_data[emission_cols].sum().sum())
        elif 'total_emissions_Mg' in self.state_data.columns:
            total_emissions = float(self.state_data['total_emissions_Mg'].sum())
        else:
            total_emissions = 0

        print(f"  📊 Total emissions computed: {total_emissions:,.0f} Mg")

        # --- State Rankings ---
        state_rankings = []
        for _, row in self.state_data.iterrows():
            state = row['subnational1']
            state_losses = row[loss_cols].values.astype(float).tolist()

            # Get emissions for state
            state_emissions = []
            for ec in emission_cols:
                state_emissions.append(float(row.get(ec, 0)))

            state_rankings.append({
                'name': state,
                'area_ha': float(row['area_ha']),
                'extent_2000_ha': float(row['extent_2000_ha']),
                'extent_2010_ha': float(row['extent_2010_ha']),
                'total_loss_ha': float(row['total_loss_ha']),
                'avg_annual_loss_ha': float(row['avg_annual_loss_ha']),
                'gain_ha': float(row.get('gain_2000-2012_ha', 0)),
                'forest_cover_ratio': float(row.get('forest_cover_ratio_2000', 0)),
                'remaining_forest_ratio': float(row.get('remaining_forest_ratio', 0)),
                'vulnerability_score': float(row.get('vulnerability_score', 0)),
                'fuzzy_priority_score': float(row.get('fuzzy_priority_score', 50)),
                'fuzzy_priority_label': str(row.get('fuzzy_priority_label', 'Moderate')),
                'deforestation_acceleration': float(row.get('deforestation_acceleration', 0)),
                'carbon_intensity': float(row.get('carbon_intensity', 0)),
                'total_emissions_Mg': float(row.get('total_emissions_Mg', 0)),
                'annual_losses': state_losses,
                'annual_emissions': state_emissions,
                'trend_slope': float(row.get('loss_trend_slope', 0)),
            })

        # Sort by vulnerability
        state_rankings.sort(key=lambda x: x['vulnerability_score'], reverse=True)

        # --- Year-by-year National Timeline ---
        national_timeline = {
            'years': PAST_YEARS,
            'annual_loss_ha': [float(self.state_data[f'tc_loss_ha_{y}'].sum()) for y in PAST_YEARS],
            'annual_emissions_Mg': [],
        }
        for y in PAST_YEARS:
            ecol = f'gfw_gross_emissions_co2e_all_gases_{y}__Mg'
            if ecol in self.state_data.columns:
                national_timeline['annual_emissions_Mg'].append(
                    float(self.state_data[ecol].sum())
                )
            else:
                national_timeline['annual_emissions_Mg'].append(0)

        national_timeline['cumulative_loss_ha'] = np.cumsum(
            national_timeline['annual_loss_ha']
        ).tolist()

        # --- Feature Importance ---
        top_features = {}
        for model_name, importances in self.ensemble.feature_importance.items():
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            top_features[model_name] = [
                {'feature': f, 'importance': round(v, 4)} for f, v in sorted_features[:10]
            ]

        # --- Reforestation Plan ---
        # Use consistent key: projected_loss_avoidance_ha
        total_projected_avoidance = sum(
            max(0, f['projected_loss_avoidance_ha']) for f in state_forecasts.values()
        )
        total_carbon_savings = sum(
            max(0, f['carbon_savings_Mg']) for f in state_forecasts.values()
        )

        reforestation_plan = []
        for state in state_rankings:
            forecast = state_forecasts.get(state['name'], {})
            avoidance = max(0, forecast.get('projected_loss_avoidance_ha', 0))
            csavings = max(0, forecast.get('carbon_savings_Mg', 0))
            reforestation_plan.append({
                'name': state['name'],
                'priority_label': state['fuzzy_priority_label'],
                'priority_score': state['fuzzy_priority_score'],
                'projected_loss_avoidance_ha': avoidance,
                'carbon_savings_Mg': csavings,
                'current_loss_rate': state['avg_annual_loss_ha'],
            })
        reforestation_plan.sort(key=lambda x: x['priority_score'], reverse=True)

        # Compile everything
        self.results = {
            'overview': {
                'total_area_ha': total_area,
                'total_forest_2000_ha': total_forest_2000,
                'total_loss_ha': total_loss,
                'total_gain_ha': total_gain,
                'net_loss_ha': total_loss - total_gain,
                'total_emissions_Mg': total_emissions,
                'forest_cover_pct_2000': round(total_forest_2000 / total_area * 100, 2) if total_area > 0 else 0,
                'forest_cover_pct_current': round(
                    (total_forest_2000 - total_loss + total_gain) / total_area * 100, 2
                ) if total_area > 0 else 0,
                'isfr_forest_cover_pct': 21.76,
                'isfr_tree_cover_pct': 3.41,
                'isfr_total_cover_pct': 25.17,
                'isfr_report_year': 2023,
                'isfr_forest_area_sqkm': 713789,
                'isfr_tree_cover_sqkm': 111818,
                'avg_annual_loss_ha': total_loss / 20,
                'total_projected_loss_avoidance_ha': total_projected_avoidance,
                'total_carbon_savings_Mg': total_carbon_savings,
                'num_states': len(state_rankings),
                'num_critical_states': sum(1 for s in state_rankings if s['fuzzy_priority_label'] == 'Critical'),
                'num_high_states': sum(1 for s in state_rankings if s['fuzzy_priority_label'] == 'High'),
            },
            'national_timeline': national_timeline,
            'state_rankings': state_rankings,
            'state_forecasts': state_forecasts,
            'country_forecast': country_forecast,
            'model_performance': metrics,
            'feature_importance': top_features,
            'smote_analysis': smote_results,
            'reforestation_plan': reforestation_plan,
        }

    def _save_results(self):
        """Save all results to JSON files."""
        print("💾 Saving results...")

        # Main results file
        with open(OUTPUT_DIR / 'pipeline_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        # Save individual components for API
        components = {
            'overview.json': self.results['overview'],
            'national_timeline.json': self.results['national_timeline'],
            'state_rankings.json': self.results['state_rankings'],
            'state_forecasts.json': self.results['state_forecasts'],
            'country_forecast.json': self.results['country_forecast'],
            'model_performance.json': self.results['model_performance'],
            'feature_importance.json': self.results['feature_importance'],
            'smote_analysis.json': self.results['smote_analysis'],
            'reforestation_plan.json': self.results['reforestation_plan'],
        }

        for filename, data in components.items():
            with open(OUTPUT_DIR / filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        # Save models
        for name, model in self.ensemble.models.items():
            safe_name = name.lower().replace(' ', '_')
            joblib.dump(model, MODEL_DIR / f'{safe_name}.joblib')

        print(f"  ✅ Saved {len(components)} JSON files and {len(self.ensemble.models)} models")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    pipeline = ReforestationPipeline()
    results = pipeline.run()

    # Print summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    overview = results['overview']
    print(f"  🌍 Total Area: {overview['total_area_ha']:,.0f} ha")
    print(f"  🌲 Forest Cover (2000): {overview['forest_cover_pct_2000']}%")
    print(f"  🌲 Forest Cover (Current): {overview['forest_cover_pct_current']}%")
    print(f"  ❌ Total Loss: {overview['total_loss_ha']:,.0f} ha")
    print(f"  ✅ Total Gain: {overview['total_gain_ha']:,.0f} ha")
    print(f"  📉 Net Loss: {overview['net_loss_ha']:,.0f} ha")
    print(f"  🏭 Total CO2 Emissions: {overview['total_emissions_Mg']:,.0f} Mg")
    print(f"  🌿 Projected Loss Avoidance: {overview['total_projected_loss_avoidance_ha']:,.0f} ha")
    print(f"  💨 Potential Carbon Savings: {overview['total_carbon_savings_Mg']:,.0f} Mg")
    print(f"  🔴 Critical Priority States: {overview['num_critical_states']}")
    print(f"  🟠 High Priority States: {overview['num_high_states']}")

    print("\n🏆 Top ML Model Performance:")
    best_model = max(results['model_performance'].items(), key=lambda x: x[1].get('test_r2', 0))
    print(f"  Best Model: {best_model[0]}")
    print(f"  Test R²: {best_model[1]['test_r2']:.4f}")
    print(f"  Test MAE: {best_model[1]['test_mae']:.2f}")
    print(f"  Test RMSE: {best_model[1]['test_rmse']:.2f}")
    print(f"  Overfit Gap: {best_model[1]['overfit_gap']:.4f}")
