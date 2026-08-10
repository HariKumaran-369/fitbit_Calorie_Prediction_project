# train several models, compare them, then tune the best one.
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from preprocessing import (
    get_clean_data,
    encode_data,
    scale_data,
    NUMERIC_COLS,
    TARGET_COL,
)
matplotlib.use("Agg")
MODEL_DIR = "models"
FIG_DIR = "reports/figures"

# hyperparameter grids for tuning the winning model.
# we only tune which model turns out to be the best one.
PARAM_GRIDS = {
    "RandomForest": {
        "n_estimators": [200, 300],
        "max_depth": [8, 10, 15],
    },
    "XGBoost": {
        "n_estimators": [200, 300],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
    },
    "Ridge": {"alpha": [0.1, 1.0, 5.0]},
    "Lasso": {"alpha": [0.01, 0.1, 1.0]},
    "KNN": {"n_neighbors": [3, 5, 7, 9]},
    "SVR": {"C": [1, 10, 50]},
}


def get_models():
    # A fresh dictionary of models (so tuning doesn't reuse fitted ones)
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "KNN": KNeighborsRegressor(n_neighbors=7),
        "DecisionTree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42
        ),
        "SVR": SVR(kernel="rbf", C=10),
        "XGBoost": XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42
        ),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Load and prepare data
    df = get_clean_data()
    df = encode_data(df)

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale the numeric columns
    num_cols = [c for c in NUMERIC_COLS if c in X_train.columns]
    X_train, scaler = scale_data(X_train, num_cols)
    X_test, _ = scale_data(X_test, num_cols, scaler)

    # Train and compare all the models
    results = []
    trained_models = {}
    models = get_models()

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        # fold cross validation to double-check the model is stable
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="r2"
        )

        results.append(
            {
                "Model": name,
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "R2": round(r2, 4),
                "CV_R2_mean": round(cv_scores.mean(), 4),
            }
        )
        trained_models[name] = model

        print(
            name,
            "-> MAE:",
            round(mae, 2),
            " RMSE:",
            round(rmse, 2),
            " R2:",
            round(r2, 4),
            " CV_R2:",
            round(cv_scores.mean(), 4),
        )

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    print("\nBest baseline model:", best_name)

    # Tune the best model with GridSearchCV (if we have a grid for it)
    if best_name in PARAM_GRIDS:
        print("Tuning", best_name, "with GridSearchCV...")
        grid = GridSearchCV(
            models[best_name], PARAM_GRIDS[best_name], cv=5, scoring="r2"
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print("Best parameters found:", grid.best_params_)
    else:
        # Nothing to tune for this model, just use it as-is
        best_model = trained_models[best_name]

    final_predictions = best_model.predict(X_test)
    final_mae = mean_absolute_error(y_test, final_predictions)
    final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
    final_r2 = r2_score(y_test, final_predictions)

    print("\nFinal tuned model:", best_name)
    print("MAE:", round(final_mae, 2))
    print("RMSE:", round(final_rmse, 2))
    print("R2:", round(final_r2, 4), "(target >= 0.80)")

    # Save charts
    plt.figure(figsize=(9, 5))
    plt.bar(results_df["Model"], results_df["R2"], color="teal")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("R2 Score")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(FIG_DIR + "/regression_model_comparison.png")
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, final_predictions, alpha=0.4, color="darkorange")
    plt.xlabel("Actual Calories Burned")
    plt.ylabel("Predicted Calories Burned")
    plt.title("Actual vs Predicted - " + best_name)
    plt.savefig(FIG_DIR + "/actual_vs_predicted.png")
    plt.close()

    if hasattr(best_model, "feature_importances_"):
        importance = pd.Series(
            best_model.feature_importances_, index=X_train.columns
        )
        importance = importance.sort_values(ascending=False).head(10)
        plt.figure(figsize=(8, 6))
        importance.plot(kind="barh")
        plt.gca().invert_yaxis()
        plt.title("Feature Importance - " + best_name)
        plt.tight_layout()
        plt.savefig(FIG_DIR + "/feature_importance.png")
        plt.close()

    # Save the best model, scaler, and results
    joblib.dump(best_model, MODEL_DIR + "/best_regression_model.pkl")
    joblib.dump(scaler, MODEL_DIR + "/regression_scaler.pkl")
    joblib.dump(
        list(X_train.columns), MODEL_DIR + "/regression_feature_columns.pkl"
    )
    results_df.to_csv("reports/regression_model_comparison.csv", index=False)

    print("\nBest model saved to models/best_regression_model.pkl")


if __name__ == "__main__":
    main()
