# REST API Reference Documentation

The Reforestation Backend exposes lightweight RESTful JSON endpoints serving historical analyses, machine learning metrics, forecasts, and prioritization tables.

All endpoints support `GET` requests and automatically return `Cache-Control: no-store` headers to ensure clients always receive up-to-date data.

---

## Base URL
- **Local Development**: `http://127.0.0.1:5000`
- **Production**: `http://<your-host>:5000`

---

## 1. Overview Statistics
- **Endpoint**: `/api/overview`
- **Method**: `GET`
- **Description**: Returns country-level aggregate metrics, official ISFR canopy benchmarks, and national totals.

### Sample Response:
```json
{
  "total_area_ha": 315279651.0,
  "total_forest_2000_ha": 38830351.0,
  "total_loss_ha": 1933252.0,
  "total_gain_ha": 255071.0,
  "net_loss_ha": 1678181.0,
  "total_emissions_Mg": 951103825.0,
  "forest_cover_pct_2000": 12.32,
  "forest_cover_pct_current": 11.78,
  "isfr_forest_cover_pct": 21.76,
  "isfr_tree_cover_pct": 3.41,
  "isfr_total_cover_pct": 25.17,
  "isfr_report_year": 2023,
  "isfr_forest_area_sqkm": 713789,
  "isfr_tree_cover_sqkm": 111818,
  "avg_annual_loss_ha": 96662.6,
  "total_projected_loss_avoidance_ha": 523529.8,
  "total_carbon_savings_Mg": 257561438.6,
  "num_states": 36,
  "num_critical_states": 3,
  "num_high_states": 13,
  "n_districts": 666,
  "ml_predicted_loss_2021_25_ha": 890159.2,
  "forecast_model": "Top-3 Blend",
  "best_cv_r2_log": 0.9349,
  "data_note": "All figures are raw GFW values at 30% canopy threshold; no scaling applied."
}
```

---

## 2. National Timeline (2001–2020)
- **Endpoint**: `/api/timeline`
- **Method**: `GET`
- **Description**: Returns year-by-year national loss, gross carbon emissions, and cumulative tree cover loss.

### Sample Response:
```json
{
  "years": [2001, 2002, 2003, ..., 2020],
  "annual_loss_ha": [62361.0, 53003.0, 47749.0, ..., 137714.0],
  "annual_emissions_Mg": [24896000.0, 20735000.0, ..., 55060000.0],
  "cumulative_loss_ha": [62361.0, 115364.0, 163113.0, ..., 1933252.0]
}
```

---

## 3. State Rankings & Vulnerability
- **Endpoint**: `/api/states`
- **Method**: `GET`
- **Description**: Returns all 36 states and union territories sorted by total forest loss, complete with vulnerability scores, acceleration indices, fuzzy priority scores, and 20-year annual loss series.

### Sample Response (Item 0):
```json
[
  {
    "name": "Mizoram",
    "area_ha": 2125175.0,
    "extent_2000_ha": 1931959.0,
    "extent_2010_ha": 1810236.0,
    "total_loss_ha": 246800.0,
    "avg_annual_loss_ha": 12340.0,
    "gain_ha": 26285.0,
    "forest_cover_ratio": 0.9091,
    "remaining_forest_ratio": 0.8205,
    "vulnerability_score": 0.8101,
    "fuzzy_priority_score": 85.5,
    "fuzzy_priority_label": "Critical",
    "deforestation_acceleration": 4.18,
    "carbon_intensity": 498.5,
    "total_emissions_Mg": 123035524.0,
    "annual_losses": [4955.0, 4465.0, ..., 24060.0],
    "annual_emissions": [2311016.0, ..., 12411001.0],
    "trend_slope": 1573.7
  }
]
```

---

## 4. Single State Detail
- **Endpoint**: `/api/state/<name>`
- **Method**: `GET`
- **Description**: Returns detailed statistics and 2021–2030 predictions for a specific state (e.g., `/api/state/Kerala`, `/api/state/Mizoram`).

---

## 5. Future Predictions & Scenarios (2021–2030)
- **Endpoint**: `/api/predictions`
- **Method**: `GET`
- **Description**: Returns both national and state-by-state 10-year projections comparing Business-As-Usual (BAU) with the Reforestation Intervention scenario.

### Sample Response:
```json
{
  "country_forecast": {
    "state": "India",
    "past": {
      "years": [2001, ..., 2020],
      "annual_loss_ha": [...]
    },
    "bau_scenario": {
      "years": [2021, ..., 2030],
      "annual_loss_ha": [156297.4, 161976.9, ..., 207412.9],
      "cumulative_loss_ha": [156297.4, 318274.3, ..., 1818552.0],
      "annual_emissions_Mg": [76893600.0, ..., 102040000.0]
    },
    "reforestation_scenario": {
      "years": [2021, ..., 2030],
      "annual_loss_ha": [148482.5, 145779.2, ..., 103706.5],
      "cumulative_loss_ha": [148482.5, ..., 1295022.0],
      "annual_emissions_Mg": [...]
    },
    "projected_loss_avoidance_ha": 523529.8,
    "carbon_savings_Mg": 257561438.6,
    "trend_slope": 5679.5
  },
  "state_forecasts": { ... }
}
```

---

## 6. Machine Learning Model Performance
- **Endpoint**: `/api/model-performance`
- **Method**: `GET`
- **Description**: Returns performance metrics for all 13 algorithms, top feature importances, and SMOTE balancing metrics.

### Sample Response:
```json
{
  "models": {
    "Top-3 Blend": {
      "train_r2": 0.9850,
      "test_r2": 0.9354,
      "train_mae": 250.0,
      "test_mae": 348.5,
      "overfit_gap": 0.0496,
      "members": ["Stacking Ensemble", "ElasticNet", "Ridge"]
    },
    "Stacking Ensemble": {
      "train_r2": 0.9880,
      "test_r2": 0.9349,
      "test_mae": 352.1,
      "overfit_gap": 0.0531
    },
    "Random Forest": {
      "train_r2": 0.9899,
      "test_r2": 0.9302,
      "test_mae": 364.3,
      "overfit_gap": 0.0597
    }
  },
  "feature_importance": {
    "Random Forest": [
      {"feature": "loss_last3_ha", "importance": 0.6196},
      {"feature": "loss_last5_ha", "importance": 0.2248}
    ]
  },
  "smote_analysis": {
    "before_smote": {
      "accuracy": 0.8410,
      "class_distribution": {"High": 194, "Medium": 332, "Low": 140}
    },
    "after_smote": {
      "accuracy": 0.9470,
      "class_distribution": {"High": 332, "Medium": 332, "Low": 332}
    }
  }
}
```

---

## 7. Strategic Reforestation Plan
- **Endpoint**: `/api/reforestation-plan`
- **Method**: `GET`
- **Description**: Returns 36 states prioritized by the Mamdani fuzzy inference engine, with target loss avoidance and carbon sequestration targets.

### Sample Response:
```json
[
  {
    "name": "Mizoram",
    "priority_label": "Critical",
    "priority_score": 85.5,
    "projected_loss_avoidance_ha": 65420.0,
    "carbon_savings_Mg": 32610500.0,
    "current_loss_rate": 28540.0
  },
  {
    "name": "Nagaland",
    "priority_label": "Critical",
    "priority_score": 82.3,
    "projected_loss_avoidance_ha": 48310.0,
    "carbon_savings_Mg": 22105400.0,
    "current_loss_rate": 20450.0
  }
]
```

---

## 8. Master Consolidated Output
- **Endpoint**: `/api/all`
- **Method**: `GET`
- **Description**: Returns the entire consolidated `pipeline_results.json` in a single API call.
