import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor

"""
    we handle machine learning here 
"""

def prepare_model_data(monthly_df: pd.DataFrame):
    df = monthly_df.copy()

    if "target_next_month_spending" not in df.columns:
        raise ValueError("Column 'target_next_month_spending' not found in monthly_df.")
    
    #drop rows with missing target
    df = df.dropna(subset=["target_next_month_spending"])

    #remove non feature columns
    X = df.drop(columns=["year_month", "target_next_month_spending"], errors="ignore")
    y = df["target_next_month_spending"]

    return X, y, df

def train_test_split_time_series(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    split_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name: str):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results = {
        "Model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4)
    }

    return results, predictions

def run_models(monthly_df: pd.DataFrame, test_size: float = 0.2):
    X, y, full_df = prepare_model_data(monthly_df)

    #reset indexes so everything stays aligned
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    full_df = full_df.reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split_time_series(X, y, test_size=test_size)

    models = {
    "Baseline Mean": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}
    results = []
    prediction_score = {}

    for model_name, model in models.items():
        metrics, preds = evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
        results.append(metrics)
        prediction_score[model_name] = preds

    results_df = pd.DataFrame(results).sort_values(by="RMSE").reset_index(drop=True)
    
    comparison_df = pd.DataFrame({
        "year_month": full_df.loc[y_test.index, "year_month"].values,
        "Actual": y_test.values
    })

    for model_name, preds in prediction_score.items():
        comparison_df[f"Predicted_{model_name}"] = preds
    
    return results_df, comparison_df

if __name__ == "__main__":
    #local test
    from preprocess import clean_transactions, create_monthly_features

    df = pd.read_csv("data/Personal_Finance_Dataset.csv")
    df = clean_transactions(df)
    monthly_df = create_monthly_features(df)

    results_df, comparison_df = run_models(monthly_df)

    print("\n Model Performance")
    print(results_df)

    print("\n Actual vs Predicted")
    print(comparison_df.head(10))