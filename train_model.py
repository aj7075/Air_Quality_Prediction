import os
import pickle
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt


DATA_PATH = "Data/city_hour.csv"
AQI_THRESHOLD = 200.0  # boundary between Normal and elevated AQI
AQI_SEVERE_THRESHOLD = 300.0  # boundary between elevated and Severe
SHAP_OUTPUT_DIR = "bi_exports"


def load_and_engineer_features() -> pd.DataFrame:
    """
    Load raw city-level hourly data and perform temporal + spatial feature engineering.

    - Keep 'City' and 'Datetime' for grouping and ordering.
    - Create lagged features (T-1, T-3, T-6 hours) for PM2.5 and NO2.
    - Create 24-hour rolling averages for all major pollutants.
    - Add simple temporal (hour, month) and spatial (city one-hot) features.
    """
    df = pd.read_csv(DATA_PATH)
    print(f"Raw data loaded from {DATA_PATH} with shape: {df.shape}")

    # Ensure required columns are present
    required_cols = {"City", "Datetime", "PM2.5", "NO2", "CO", "SO2", "O3", "AQI"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # Parse datetime and sort for temporal operations
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values(["City", "Datetime"]).reset_index(drop=True)

    # Basic filtering: drop rows without target AQI
    df = df[~df["AQI"].isna()].copy()

    # Forward-fill pollutants within each city to reduce missingness
    pollutant_cols = ["PM2.5", "NO2", "CO", "SO2", "O3"]
    df[pollutant_cols] = (
        df.groupby("City")[pollutant_cols]
        .apply(lambda g: g.ffill().bfill())
        .reset_index(level=0, drop=True)
    )

    # Lagged features for PM2.5 and NO2
    for pollutant in ["PM2.5", "NO2"]:
        for lag_h in [1, 3, 6]:
            df[f"{pollutant}_lag{lag_h}"] = (
                df.groupby("City")[pollutant].shift(lag_h)
            )

    # 24-hour rolling averages for major pollutants
    for pollutant in pollutant_cols:
        df[f"{pollutant}_roll24"] = (
            df.groupby("City")[pollutant]
            .rolling(window=24, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # Simple temporal features
    df["hour"] = df["Datetime"].dt.hour
    df["month"] = df["Datetime"].dt.month

    # Drop rows where engineered features are still missing
    engineered_cols = [
        "PM2.5_lag1",
        "PM2.5_lag3",
        "PM2.5_lag6",
        "NO2_lag1",
        "NO2_lag3",
        "NO2_lag6",
    ] + [f"{p}_roll24" for p in pollutant_cols]

    df = df.dropna(subset=pollutant_cols + engineered_cols + ["AQI"])

    # Spatial features: one-hot encode City (drop first to avoid collinearity)
    city_dummies = pd.get_dummies(df["City"], prefix="City", drop_first=True)
    df = pd.concat([df, city_dummies], axis=1)

    # Multi-class risk bucket for Stage 1 classifier
    # 0: Normal (<200), 1: Moderate-High (200-300), 2: Severe (>300)
    df["risk_bucket"] = np.select(
        [
            df["AQI"] < AQI_THRESHOLD,
            (df["AQI"] >= AQI_THRESHOLD) & (df["AQI"] <= AQI_SEVERE_THRESHOLD),
            df["AQI"] > AQI_SEVERE_THRESHOLD,
        ],
        [0, 1, 2],
        default=0,
    ).astype(int)

    print(f"After feature engineering, data shape: {df.shape}")
    return df


def build_feature_matrix(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build feature matrix X and targets for both classification and regression.
    """
    base_features = ["PM2.5", "NO2", "CO", "SO2", "O3"]
    lag_features = [
        "PM2.5_lag1",
        "PM2.5_lag3",
        "PM2.5_lag6",
        "NO2_lag1",
        "NO2_lag3",
        "NO2_lag6",
    ]
    roll_features = [f"{p}_roll24" for p in base_features]
    temporal_features = ["hour", "month"]
    city_features = [c for c in df.columns if c.startswith("City_")]

    feature_cols = (
        base_features + lag_features + roll_features + temporal_features + city_features
    )

    X = df[feature_cols].copy()
    y_reg = df["AQI"].astype(float)
    y_cls = df["risk_bucket"].astype(int)

    print(f"Total number of features: {len(feature_cols)}")
    print("Feature columns:", feature_cols)

    return {
        "X": X,
        "y_reg": y_reg,
        "y_cls": y_cls,
        "feature_cols": feature_cols,
    }


def train_hybrid_models(
    X: pd.DataFrame, y_reg: pd.Series, y_cls: pd.Series, feature_cols
) -> Dict[str, Any]:
    """
    Train the hybrid cascade system:

    - Stage 1: RandomForestClassifier for multi-class regime gating:
        * 0: Normal (AQI < 200)
        * 1: Moderate-High (200 <= AQI <= 300)
        * 2: Severe (AQI > 300)
    - Stage 2: three specialized XGBoost regressors (one per regime).
      Severe regressor uses AQI-dependent sample weighting to emphasize extreme days.
    - Baseline: RandomForestRegressor on full data (for backward compatibility).
    """
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X,
        y_reg,
        y_cls,
        test_size=0.1,  # 10% test, 90% train
        random_state=42,
        stratify=y_cls,
    )

    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    # Stage 1: multi-class classifier
    stage1_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    stage1_clf.fit(X_train, y_cls_train)
    print("Stage 1 classifier trained.")
    print("Classification report (Stage 1) on test data:")
    print(classification_report(y_cls_test, stage1_clf.predict(X_test)))

    # Split training data into three regimes for Stage 2
    normal_mask = y_reg_train < AQI_THRESHOLD
    mid_mask = (y_reg_train >= AQI_THRESHOLD) & (y_reg_train <= AQI_SEVERE_THRESHOLD)
    severe_mask = y_reg_train > AQI_SEVERE_THRESHOLD

    X_train_normal = X_train[normal_mask]
    y_train_normal = y_reg_train[normal_mask]
    X_train_mid = X_train[mid_mask]
    y_train_mid = y_reg_train[mid_mask]
    X_train_severe = X_train[severe_mask]
    y_train_severe = y_reg_train[severe_mask]

    print(
        f"Normal samples: {X_train_normal.shape[0]}, "
        f"Mid (200-300) samples: {X_train_mid.shape[0]}, "
        f"Severe (>300) samples: {X_train_severe.shape[0]}"
    )

    # Stage 2a: Normal regime regressor
    stage2_normal = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )
    stage2_normal.fit(X_train_normal, y_train_normal)
    print("Stage 2 (Normal) XGBoost regressor trained.")

    # Stage 2b: Mid regime regressor (200-300)
    if X_train_mid.shape[0] > 0:
        stage2_mid = XGBRegressor(
            n_estimators=600,
            learning_rate=0.04,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
        )
        stage2_mid.fit(X_train_mid, y_train_mid)
        print("Stage 2 (Mid 200-300) XGBoost regressor trained.")
    else:
        stage2_mid = None
        print("Warning: No Mid (200-300) samples found for Stage 2 mid regressor.")

    # Stage 2c: Severe regime regressor with heavy sample weighting
    if X_train_severe.shape[0] > 0:
        y_sev_centered = np.maximum(y_train_severe - AQI_SEVERE_THRESHOLD, 0.0)
        if y_sev_centered.max() > 0:
            weights_sev = 1.0 + 4.0 * (y_sev_centered / y_sev_centered.max())
        else:
            weights_sev = np.ones_like(y_sev_centered)

        stage2_severe = XGBRegressor(
            n_estimators=700,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
        )
        stage2_severe.fit(X_train_severe, y_train_severe, sample_weight=weights_sev)
        print("Stage 2 (Severe >300) XGBoost regressor trained with sample weighting.")
    else:
        stage2_severe = None
        print("Warning: No Severe (>300) samples found for Stage 2 severe regressor.")

    # Baseline RandomForestRegressor on full data (kept for backward compatibility)
    baseline_rf = RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    baseline_rf.fit(X_train, y_reg_train)
    print("Baseline RandomForestRegressor trained on full data.")

    # Evaluate cascade on test set
    stage1_preds = stage1_clf.predict(X_test)
    normal_test_mask = stage1_preds == 0
    mid_test_mask = stage1_preds == 1
    severe_test_mask = stage1_preds == 2

    y_pred_cascade = np.zeros_like(y_reg_test.values, dtype=float)
    if normal_test_mask.any():
        y_pred_cascade[normal_test_mask] = stage2_normal.predict(X_test[normal_test_mask])

    if mid_test_mask.any():
        if stage2_mid is not None:
            y_pred_cascade[mid_test_mask] = stage2_mid.predict(X_test[mid_test_mask])
        else:
            y_pred_cascade[mid_test_mask] = stage2_normal.predict(X_test[mid_test_mask])

    if severe_test_mask.any():
        if stage2_severe is not None:
            y_pred_cascade[severe_test_mask] = stage2_severe.predict(X_test[severe_test_mask])
        else:
            # fallback: use mid if available else normal
            fallback = stage2_mid if stage2_mid is not None else stage2_normal
            y_pred_cascade[severe_test_mask] = fallback.predict(X_test[severe_test_mask])

    mse = mean_squared_error(y_reg_test, y_pred_cascade)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_reg_test, y_pred_cascade)
    print(f"Hybrid cascade RMSE on test: {rmse:.3f}")
    print(f"Hybrid cascade R^2 on test: {r2:.4f}")

    return {
        "model": baseline_rf,  # for existing prediction pipeline
        "stage1_classifier": stage1_clf,
        "stage2_normal_regressor": stage2_normal,
        "stage2_mid_regressor": stage2_mid,
        "stage2_severe_regressor": stage2_severe,
        "feature_columns": feature_cols,
        "aqi_threshold": AQI_THRESHOLD,
        "aqi_severe_threshold": AQI_SEVERE_THRESHOLD,
    }


def generate_shap_summary(
    model: XGBRegressor, X_sample: pd.DataFrame, label: str
) -> None:
    """
    Generate and save a SHAP summary plot for a given XGBoost model.
    """
    if not os.path.exists(SHAP_OUTPUT_DIR):
        os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)

    print(f"Computing SHAP values for {label} model on {X_sample.shape[0]} samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    output_path = os.path.join(SHAP_OUTPUT_DIR, f"shap_summary_{label}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"SHAP summary plot saved to {output_path}")


def main():
    try:
        df = load_and_engineer_features()
        matrices = build_feature_matrix(df)
        X, y_reg, y_cls, feature_cols = (
            matrices["X"],
            matrices["y_reg"],
            matrices["y_cls"],
            matrices["feature_cols"],
        )

        artifacts = train_hybrid_models(X, y_reg, y_cls, feature_cols)

        # Generate SHAP plots focused on chemical drivers of AQI
        # Use samples from Normal and Severe regimes (more stable for discussion)
        normal_mask = y_reg < AQI_THRESHOLD
        severe_mask = y_reg > AQI_SEVERE_THRESHOLD

        X_normal = X[normal_mask]
        X_severe = X[severe_mask]

        # Subsample for efficiency
        X_normal_sample = X_normal.sample(n=min(2000, len(X_normal)), random_state=42)
        X_severe_sample = (
            X_severe.sample(n=min(2000, len(X_severe)), random_state=42)
            if len(X_severe) > 0
            else None
        )

        if isinstance(artifacts["stage2_normal_regressor"], XGBRegressor):
            generate_shap_summary(
                artifacts["stage2_normal_regressor"],
                X_normal_sample,
                label="normal_regime",
            )
        if (
            artifacts.get("stage2_severe_regressor") is not None
            and X_severe_sample is not None
            and isinstance(artifacts["stage2_severe_regressor"], XGBRegressor)
        ):
            generate_shap_summary(
                artifacts["stage2_severe_regressor"],
                X_severe_sample,
                label="severe_regime",
            )

        # Persist artifacts, including baseline model for compatibility
        with open("./new_model.pkl", "wb") as file:
            pickle.dump(artifacts, file)

        print("Hybrid temporal-spatial model trained and saved successfully.")
        print("Artifacts keys:", list(artifacts.keys()))

    except Exception as e:
        print(f"Error during training pipeline: {e}")


if __name__ == "__main__":
    main()