## Project Overview (Research Version)

This project predicts **Air Quality Index (AQI)** from pollutant measurements and demonstrates a **research-grade Hybrid Temporal–Spatial Predictive System** designed to handle **high-pollution (high AQI) regimes** and provide **explainability (XAI)** for research reporting.

---

## What the Project Does

- **AQI prediction** from pollutant components (manual input in Streamlit).
- **Geo-location monitoring** and **India AQI map** (WAQI-based) for visualization and live context.
- **Research module**: a **hybrid cascade** that uses temporal + spatial features and regime-specific regressors.
- **Explainability (SHAP)**: feature-importance summary plots for the hybrid model (Normal vs High-Risk regimes).

---

## Datasets You Have

### Raw temporal–spatial dataset (used for research model)

- **File**: `Data/city_hour.csv`
- **Original size**: ~707,875 rows × 16 columns
- **What it contains**:
  - `City`, `Datetime` (hourly timestamps)
  - Pollutants (examples): `PM2.5`, `PM10`, `NO`, `NO2`, `NOx`, `NH3`, `CO`, `SO2`, `O3`, `Benzene`, `Toluene`, `Xylene`
  - Target: `AQI` (+ optional label `AQI_Bucket`)

### Cleaned tabular dataset (used for baseline/ANN experiments)

- **File**: `Data/final_data.csv`
- **Size**: ~22,610 rows × 6 columns
- **Columns**: `PM2.5, NO2, CO, SO2, O3, AQI`
- **Limitation**: contains **no** `City` or `Datetime`, so it cannot represent temporal trends or spatial heterogeneity.

---

## Units of Components (Pollutants)

These are the standard units used in the CPCB-style datasets and are the intended interpretation for your columns:

- **PM2.5, PM10**: \( \mu g/m^3 \)
- **NO, NO2, NOx, NH3, SO2, O3**: \( \mu g/m^3 \)
- **CO**: \( mg/m^3 \)
- **Benzene, Toluene, Xylene**: \( \mu g/m^3 \)
- **AQI**: **unitless index** (0–500+ scale)

---

## Code Modules (What Each File Does)

### Application (Streamlit)

- `app.py`
  - Streamlit entry point and menu routing.
- `prediction.py`
  - Manual AQI prediction UI.
  - Geo-location AQI page and India AQI map.
  - **Hybrid model UI toggle** (Baseline RF vs Hybrid Cascade).
- `explore_page.py`
  - Data exploration visuals for the cleaned dataset (charts/plots).

### Research Training + Evaluation

- `train_model.py`
  - Trains the research-grade hybrid pipeline using `city_hour.csv`.
  - Saves trained artifacts to `new_model.pkl`.
  - Generates SHAP summary plots for Normal and High-Risk regimes.
- `evaluate_models.py`
  - Trains baseline + hybrid (same split), prints overall + band-wise test scores.
- `generate_bi_exports.py` + `bi_exports/`
  - Exports evaluation tables and analysis artifacts for reporting/BI usage.

### Notebooks

- `models/7. Implementing ANN.ipynb`
  - ANN baseline + **custom penalty-weighted loss** that prioritizes high AQI.

---

## Research Contribution: Hybrid Temporal–Spatial Cascading

### Why a hybrid cascade?

In AQI prediction, the relationship between pollutants and AQI behaves differently in:

- **Normal / moderate conditions** (AQI < 200), where relationships are smoother and error tolerance is higher.
- **High-risk / severe conditions** (AQI ≥ 200 and especially > 300), where:
  - extreme values are rarer,
  - standard models often underperform due to imbalance and regime shift,
  - prediction errors are more costly for health-risk messaging.

So the project implements a **two-stage cascade**:

1. **Stage 1 Classifier (Random Forest Classifier)**
   - Predicts regime: **Normal** (AQI < 200) vs **High-Risk** (AQI ≥ 200)
2. **Stage 2 Specialized Regressors (XGBoost)**
   - **Normal regressor** trained only on Normal samples
   - **High-Risk regressor** trained only on High-Risk samples with **AQI-dependent weighting**

This is not stacking (output-as-input); Stage 1 acts as a **gating model** that selects the right regressor.

---

## Temporal Feature Engineering (Research Model)

Using `Data/city_hour.csv`, the pipeline:

- **Keeps** `City` and `Datetime` until features are created.
- Builds per-city time-ordered sequences.
- Adds:
  - **Lag features** (T−1, T−3, T−6 hours) for `PM2.5` and `NO2`
  - **24-hour rolling mean** for `PM2.5, NO2, CO, SO2, O3`
  - Time indicators: `hour`, `month`
  - Spatial indicators: **one-hot encoded** `City_*` columns

---

## Training / Testing Split (Chosen for Paper)

- **Split**: **90% train / 10% test**
- **Method**: `train_test_split(..., stratify=risk_bucket, random_state=42)`
  - Stratification ensures both Normal and High-Risk regimes are represented in train and test.

> Note: changing 80/20 vs 90/10 does not fix model weaknesses by itself. It mainly changes how much data is available for training and how stable the test estimate is. The hybrid model’s weakness is concentrated in the 200–300 band and needs modeling/tuning, not only split changes.

---

## Model Evaluation Results (90/10 Split)

These are the latest **test set** results (10% holdout, ~57,530 samples):

### Stage 1 (Classifier) – Normal vs High-Risk

- **Accuracy**: ~0.98
- Normal: precision ~0.98, recall ~0.99
- High-Risk: precision ~0.98, recall ~0.95

### Baseline Random Forest Regressor (single-stage)

- **Overall**: RMSE **34.919**, R² **0.9537**
- **Band-wise**:
  - Normal (AQI < 200): RMSE **13.464**, R² **0.893**
  - Moderate-High (200–300): RMSE **25.927**, R² **0.210**
  - Severe (AQI > 300): RMSE **82.300**, R² **0.906**

### Hybrid Cascade (Stage 1 + two XGBoost regressors)

- **Overall**: RMSE **37.422**, R² **0.9468**
- **Band-wise**:
  - Normal (AQI < 200): RMSE **17.800**, R² **0.814**
  - Moderate-High (200–300): RMSE **40.442**, R² **−0.923**
  - Severe (AQI > 300): RMSE **81.084**, R² **0.909**

**Interpretation**:

- The **baseline RF** currently has the lowest overall error.
- The **hybrid model** is comparable in the **Severe** band but underperforms in the **200–300** band, which indicates the cascade boundary/weighting needs tuning for mid-high AQI.

---

## Explainability (XAI) with SHAP

For research reporting, the hybrid model generates SHAP summary plots:

- `bi_exports/shap_summary_normal_regime.png`
- `bi_exports/shap_summary_high_risk_regime.png`

These plots are used in the discussion section to explain **chemical drivers of AQI** and how feature importance shifts between Normal and High-Risk conditions.

---

## Frontend Demo (Panel-Ready)

The Streamlit app supports showing research progress:

- **Predict page** includes a toggle:
  - **Baseline Random Forest**
  - **Hybrid Temporal–Spatial Cascade (Research)**

This lets you demonstrate that the project now supports a research hybrid pipeline beyond the earlier single-model RF demo.

---

## How to Reproduce (Commands)

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the research model (writes `new_model.pkl` and SHAP plots):

```bash
python train_model.py
```

Evaluate baseline vs hybrid (prints metrics):

```bash
python evaluate_models.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```

