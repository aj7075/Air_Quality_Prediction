"""
Model evaluation script for the Air Quality Prediction project.

This script:
- Reuses the temporal–spatial feature engineering from `train_model.py`.
- Trains:
  - A baseline RandomForestRegressor (single-stage).
  - A hybrid cascade model:
      * Stage 1 RandomForestClassifier (Normal vs High-Risk).
      * Stage 2 XGBoost regressors for Normal and High-Risk regimes.
- Evaluates both models on a held-out test set with:
  - Overall RMSE and R².
  - Band-wise metrics in different AQI regimes:
      * Normal: AQI < 200
      * Moderate-High: 200 ≤ AQI ≤ 300
      * Severe: AQI > 300

Run:
    python evaluate_models.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from train_model import (
    load_and_engineer_features,
    build_feature_matrix,
    AQI_THRESHOLD,
    AQI_SEVERE_THRESHOLD,
)


def train_baseline_rf(X_train, y_train):
    """Train a baseline RandomForestRegressor on all regimes."""
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_hybrid_cascade(X_train, y_reg_train, y_cls_train):
    """
    Train the hybrid cascade model (3-regime):

    - Stage 1 classifier: RF multi-class gating:
        0: AQI < 200
        1: 200 <= AQI <= 300
        2: AQI > 300
    - Stage 2 regressors:
        - Normal regressor (AQI < 200)
        - Mid regressor (200-300)
        - Severe regressor (>300) with AQI-dependent sample weighting
    """
    # Stage 1 classifier
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_cls_train)

    # Split into regimes for Stage 2
    normal_mask = y_reg_train < AQI_THRESHOLD
    mid_mask = (y_reg_train >= AQI_THRESHOLD) & (y_reg_train <= AQI_SEVERE_THRESHOLD)
    severe_mask = y_reg_train > AQI_SEVERE_THRESHOLD

    X_train_normal = X_train[normal_mask]
    y_train_normal = y_reg_train[normal_mask]
    X_train_mid = X_train[mid_mask]
    y_train_mid = y_reg_train[mid_mask]
    X_train_severe = X_train[severe_mask]
    y_train_severe = y_reg_train[severe_mask]

    # Stage 2 Normal regressor
    reg_normal = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )
    reg_normal.fit(X_train_normal, y_train_normal)

    # Stage 2 Mid regressor (200-300)
    if X_train_mid.shape[0] > 0:
        reg_mid = XGBRegressor(
            n_estimators=600,
            learning_rate=0.04,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
        )
        reg_mid.fit(X_train_mid, y_train_mid)
    else:
        reg_mid = None

    # Stage 2 Severe regressor (>300) with AQI-weighted samples
    if X_train_severe.shape[0] > 0:
        y_sev_centered = np.maximum(y_train_severe - AQI_SEVERE_THRESHOLD, 0.0)
        if y_sev_centered.max() > 0:
            weights_sev = 1.0 + 4.0 * (y_sev_centered / y_sev_centered.max())
        else:
            weights_sev = np.ones_like(y_sev_centered)

        reg_severe = XGBRegressor(
            n_estimators=700,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
        )
        reg_severe.fit(X_train_severe, y_train_severe, sample_weight=weights_sev)
    else:
        reg_severe = None

    return clf, reg_normal, reg_mid, reg_severe


def predict_cascade(clf, reg_normal, reg_mid, reg_severe, X):
    """Run the hybrid cascade on feature matrix X."""
    stage1_preds = clf.predict(X)
    normal_mask = stage1_preds == 0
    mid_mask = stage1_preds == 1
    severe_mask = stage1_preds == 2

    y_pred = np.zeros(X.shape[0], dtype=float)

    if normal_mask.any():
        y_pred[normal_mask] = reg_normal.predict(X[normal_mask])

    if mid_mask.any():
        if reg_mid is not None:
            y_pred[mid_mask] = reg_mid.predict(X[mid_mask])
        else:
            y_pred[mid_mask] = reg_normal.predict(X[mid_mask])

    if severe_mask.any():
        if reg_severe is not None:
            y_pred[severe_mask] = reg_severe.predict(X[severe_mask])
        else:
            fallback = reg_mid if reg_mid is not None else reg_normal
            y_pred[severe_mask] = fallback.predict(X[severe_mask])

    return y_pred


def bandwise_metrics(y_true, y_pred):
    """Compute RMSE and R² in AQI bands."""
    bands = [
        ("Normal (AQI < 200)", 0, 200),
        ("Moderate-High (200 <= AQI <= 300)", 200, 300),
        ("Severe (AQI > 300)", 300, None),
    ]

    rows = []
    for name, low, high in bands:
        if high is None:
            mask = y_true > low
        else:
            mask = (y_true >= low) & (y_true <= high)

        if mask.sum() == 0:
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]
        mse = mean_squared_error(y_t, y_p)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_t, y_p)
        rows.append(
            {
                "Band": name,
                "n_samples": int(mask.sum()),
                "RMSE": rmse,
                "R2": r2,
            }
        )

    return pd.DataFrame(rows)


def main():
    # 1. Load and engineer temporal–spatial features
    df = load_and_engineer_features()
    matrices = build_feature_matrix(df)
    X = matrices["X"]
    y_reg = matrices["y_reg"]
    y_cls = matrices["y_cls"]

    # 2. Train/test split (90% train, 10% test)
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X,
        y_reg,
        y_cls,
        test_size=0.1,
        random_state=42,
        stratify=y_cls,
    )

    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    # 3. Train baseline RF
    baseline_rf = train_baseline_rf(X_train, y_reg_train)

    # 4. Train hybrid cascade
    clf, reg_normal, reg_mid, reg_severe = train_hybrid_cascade(
        X_train, y_reg_train, y_cls_train
    )

    # 5. Evaluate Stage 1 classifier
    print("\n=== Stage 1 Classifier Evaluation (Test Set) ===")
    y_cls_pred = clf.predict(X_test)
    print(
        classification_report(
            y_cls_test,
            y_cls_pred,
            labels=[0, 1, 2],
            target_names=["Normal", "Moderate-High", "Severe"],
        )
    )

    # 6. Evaluate baseline RF on test set
    y_pred_baseline = baseline_rf.predict(X_test)
    mse_base = mean_squared_error(y_reg_test, y_pred_baseline)
    rmse_base = np.sqrt(mse_base)
    r2_base = r2_score(y_reg_test, y_pred_baseline)

    print("\n=== Baseline RF (All Regimes) – Test Set ===")
    print(f"RMSE: {rmse_base:.3f}")
    print(f"R²  : {r2_base:.4f}")

    print("\nBand-wise metrics (Baseline RF):")
    df_base_bands = bandwise_metrics(y_reg_test.values, y_pred_baseline)
    print(df_base_bands.to_string(index=False))

    # 7. Evaluate hybrid cascade on test set
    y_pred_cascade = predict_cascade(clf, reg_normal, reg_mid, reg_severe, X_test)
    mse_cascade = mean_squared_error(y_reg_test, y_pred_cascade)
    rmse_cascade = np.sqrt(mse_cascade)
    r2_cascade = r2_score(y_reg_test, y_pred_cascade)

    print("\n=== Hybrid Cascade – Test Set ===")
    print(f"RMSE: {rmse_cascade:.3f}")
    print(f"R²  : {r2_cascade:.4f}")

    print("\nBand-wise metrics (Hybrid Cascade):")
    df_cascade_bands = bandwise_metrics(y_reg_test.values, y_pred_cascade)
    print(df_cascade_bands.to_string(index=False))


if __name__ == "__main__":
    main()

