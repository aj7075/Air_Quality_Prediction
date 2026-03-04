## Project Overview

This project builds an Air Quality Index (AQI) prediction system that combines classical machine learning and deep learning with temporal–spatial information. The goal is to estimate AQI values from pollutant measurements, capture how air quality evolves over time and across cities, and explain which pollutants drive AQI, especially on high‑pollution (Severe) days.

At a high level, the system:
- Predicts numeric AQI values from pollutant concentrations.
- Uses temporal context (hourly history, rolling averages) and spatial context (city) to improve accuracy.
- Places special emphasis on correctly modeling high‑AQI regimes that are usually under‑represented and more difficult to predict.
- Provides explainability (via SHAP) so that the chemical drivers of AQI can be discussed in a research setting.

---

## Data Sources

The project uses two main data files in the `Data/` directory:

- `Data/city_hour.csv` – raw, rich temporal–spatial dataset.
- `Data/final_data.csv` – cleaned, compact tabular dataset used for baseline models.

### 1. Raw Temporal–Spatial Dataset: `Data/city_hour.csv`

This is the primary, information‑rich dataset used for the hybrid temporal–spatial model.

- **Granularity**: Hourly measurements.
- **Spatial coverage**: Multiple Indian cities (e.g., Ahmedabad, Delhi, etc.).
- **Time dimension**: Several years of hourly observations per city.

**Key columns:**

- **Metadata / context**
  - `City`: Name of the city where the measurement was taken.
  - `Datetime`: Timestamp of the observation (`YYYY-MM-DD HH:MM:SS`).

- **Pollutant measurements (inputs/features)**
  - `PM2.5`: Fine particulate matter (µg/m³).
  - `PM10`: Coarse particulate matter (µg/m³).
  - `NO`: Nitric oxide.
  - `NO2`: Nitrogen dioxide.
  - `NOx`: Total nitrogen oxides.
  - `NH3`: Ammonia.
  - `CO`: Carbon monoxide.
  - `SO2`: Sulfur dioxide.
  - `O3`: Ozone.
  - `Benzene`, `Toluene`, `Xylene`: Volatile organic compounds (VOCs).

- **Targets / labels**
  - `AQI`: Numerical Air Quality Index value.
  - `AQI_Bucket`: Categorical AQI label (e.g., Good, Satisfactory, Moderate, Poor, Very Poor, Severe), often partially populated in the raw file.

**Why this dataset is important:**

- It provides both **temporal** information (hourly progression) and **spatial** information (city‑level differences).
- It enables temporal feature engineering such as:
  - Lagged features at T‑1, T‑3, and T‑6 hours for key pollutants (e.g., `PM2.5`, `NO2`).
  - 24‑hour rolling averages for major pollutants (`PM2.5`, `CO`, `SO2`, `O3`, `NO2`).
  - Time‑of‑day (`hour`) and seasonal (`month`) indicators.
- It supports spatial feature engineering:
  - City‑level one‑hot encodings to capture persistent differences between locations.

These properties make `city_hour.csv` the natural foundation for a research‑grade, hybrid temporal–spatial predictive system.

### 2. Cleaned Tabular Dataset: `Data/final_data.csv`

This dataset is a preprocessed, compact version that is convenient for baseline modeling and experiments, especially for models that do not explicitly use time or city.

- **Columns:**
  - `PM2.5`
  - `NO2`
  - `CO`
  - `SO2`
  - `O3`
  - `AQI`

Here, each row consists of:

- A 5‑dimensional feature vector:
  - `PM2.5` (µg/m³),
  - `NO2` (µg/m³),
  - `CO` (often mg/m³ or ppm‑equivalent),
  - `SO2` (µg/m³),
  - `O3` (µg/m³),
- and a scalar target:
  - `AQI` (numeric air quality index).

**What this dataset omits:**

- No `City` information.
- No `Datetime` (no explicit temporal ordering).
- No additional pollutants such as `PM10`, `NOx`, `NH3`, or VOCs.

As a result, `final_data.csv` is ideal for:

- Simple, static models (e.g., basic Random Forest regression, early ANN experiments).
- Quick benchmarking of model architectures without the extra complexity of time and space.

But it is less suitable when:

- We need to capture **temporal dynamics** (pollution trends over hours/days).
- We care about **city‑specific behavior** and spatial heterogeneity.
- We want to emphasize performance on rare but extreme **high‑AQI (Severe)** events that depend on build‑up over time.

For those reasons, the current research‑oriented pipeline uses `city_hour.csv` as the master data source and treats `final_data.csv` as a convenient, derived dataset for baseline comparisons.

---

## Relationship Between Data and Models

The project currently uses these datasets in two main ways:

1. **Baseline models (Random Forest, ANN)**
   - Input: `Data/final_data.csv`.
   - Features: `[PM2.5, NO2, CO, SO2, O3]`.
   - Target: `AQI`.
   - Pros: Simple, fast to train, easy to interpret.
   - Limitation: No explicit temporal or spatial context, which can lead to overfitting and poor generalization in the high‑AQI regime (e.g., AQI > 300).

2. **Hybrid temporal–spatial model**
   - Input: `Data/city_hour.csv`.
   - Uses `City` and `Datetime` to construct:
     - Lagged features (T‑1, T‑3, T‑6 hours) for `PM2.5` and `NO2`.
     - 24‑hour rolling averages for `PM2.5`, `CO`, `SO2`, `O3`, `NO2`.
     - Temporal indicators: `hour`, `month`.
     - Spatial indicators: one‑hot encoded city features.
   - Target: `AQI`, with an additional derived label:
     - `risk_bucket` = 0 for AQI < 200 (Normal), 1 for AQI ≥ 200 (High‑Risk).

This richer representation allows the hybrid system to:

- Explicitly separate **Normal** and **High‑Risk** regimes.
- Train specialized regressors for each regime.
- Study the chemical drivers of AQI under different conditions, using SHAP‑based explainability on the engineered feature space.

---

## Summary

- The project is an **Air Quality Prediction** system focused on estimating AQI from pollutant measurements and understanding the drivers of air quality across time and space.
- You have:
  - A **raw, temporal–spatial dataset** (`city_hour.csv`) with rich pollutant, city, and timestamp information.
  - A **cleaned, compact dataset** (`final_data.csv`) with five core pollutants and AQI, suitable for baseline models.
- The current research direction leverages `city_hour.csv` to build lagged and rolling features, incorporate temporal and spatial context, and train advanced hybrid models that emphasize performance on extreme, high‑AQI events.

