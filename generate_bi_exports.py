import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None  # type: ignore


DATA_PATH = os.path.join("Data", "final_data.csv")
OUTPUT_DIR = "bi_exports"


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "AQI" not in df.columns:
        raise ValueError("Expected target column 'AQI' not found in dataset.")

    return df


def train_models(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c != "AQI"]
    X = df[feature_cols]
    y = df["AQI"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models: Dict[str, object] = {}

    # Baseline: simple Linear Regression
    models["Linear Regression (Baseline)"] = LinearRegression()

    # Random Forest
    models["Random Forest"] = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )

    # XGBoost (if available)
    if XGBRegressor is not None:
        models["XGBoost (Final)"] = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
        )

    metrics_rows: List[Dict] = []
    errors_rows: List[Dict] = []
    feature_importance_rows: List[Dict] = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        phase = "before" if "Baseline" in name or "Linear" in name else "after"

        metrics_rows.append(
            {
                "model_name": name,
                "phase": phase,
                "r2": r2,
                "mse": mse,
                "rmse": rmse,
            }
        )

        # store per-sample errors for Power BI
        for actual, pred in zip(y_test.values, y_pred):
            errors_rows.append(
                {
                    "model_name": name,
                    "actual_aqi": float(actual),
                    "predicted_aqi": float(pred),
                    "error": float(pred - actual),
                }
            )

        # feature importance / coefficients
        if hasattr(model, "feature_importances_"):
            importances = np.array(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_)
            importances = coefs / (coefs.sum() + 1e-9)
        else:
            importances = None

        if importances is not None:
            for feat, imp in zip(feature_cols, importances):
                feature_importance_rows.append(
                    {
                        "model_name": name,
                        "feature": feat,
                        "importance": float(imp),
                        "phase": phase,
                    }
                )

    return metrics_rows, errors_rows, feature_importance_rows


def write_improvements_template(path: str):
    improvements = [
        {
            "area": "Modeling",
            "before": "Simple baseline Linear Regression with limited tuning.",
            "after": "Ensemble models (Random Forest / XGBoost) with better generalization.",
            "impact_type": "Accuracy",
            "impact_level": "High",
        },
        {
            "area": "Data Handling",
            "before": "Basic use of pollutant features without rich exploration.",
            "after": "Cleaned dataset with selected key pollutants and better preprocessing.",
            "impact_type": "Reliability",
            "impact_level": "Medium",
        },
        {
            "area": "User Experience",
            "before": "Basic AQI prediction interface only.",
            "after": "Streamlit app with geolocation, maps, health and psychological recommendations.",
            "impact_type": "User Experience",
            "impact_level": "High",
        },
    ]
    df_imp = pd.DataFrame(improvements)
    df_imp.to_csv(path, index=False)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()
    metrics_rows, errors_rows, feature_importance_rows = train_models(df)

    # Model comparison
    model_comp_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    pd.DataFrame(metrics_rows).to_csv(model_comp_path, index=False)

    # Error analysis
    error_path = os.path.join(OUTPUT_DIR, "error_analysis.csv")
    pd.DataFrame(errors_rows).to_csv(error_path, index=False)

    # Feature importance
    fi_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    pd.DataFrame(feature_importance_rows).to_csv(fi_path, index=False)

    # Manual improvements table template
    improvements_path = os.path.join(OUTPUT_DIR, "improvements_template.csv")
    write_improvements_template(improvements_path)

    # Optionally save the strongest model for reference
    best_row = max(metrics_rows, key=lambda r: r["r2"])
    best_model_name = best_row["model_name"]

    print("BI export files generated in:", OUTPUT_DIR)
    print("Best model by R2:", best_model_name, "R2 =", best_row["r2"])


if __name__ == "__main__":
    main()


