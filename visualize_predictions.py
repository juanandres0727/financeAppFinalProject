import matplotlib.pyplot as plt
import pandas as pd


def plot_actual_vs_predicted(comparison_df: pd.DataFrame, prediction_col: str):
    plt.figure(figsize=(12, 6))
    plt.plot(comparison_df["year_month"], comparison_df["Actual"], marker="o", label="Actual")
    plt.plot(comparison_df["year_month"], comparison_df[prediction_col], marker="o", label=prediction_col)

    plt.title("Actual vs Predicted Monthly Spending")
    plt.xlabel("Month")
    plt.ylabel("Spending")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()