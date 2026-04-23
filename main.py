import pandas as pd
from preprocess import clean_transactions, create_monthly_features
from model import run_models
from visualize_predictions import plot_actual_vs_predicted

"""
    This file runs the project:
        - we load the dataset, clean it and create monthly features
        - we run EDA, train models and print results 
"""

if __name__ == "__main__":
    df = pd.read_csv("data/Personal_Finance_Dataset.csv")
    df = clean_transactions(df)
    monthly_df = create_monthly_features(df)

    results_df, comparison_df = run_models(monthly_df)

    print("\nModel Results")
    print(results_df)

    print("\nPredictions Preview")
    print(comparison_df.head())

from eda import run_eda
from model import run_models

if __name__ == "__main__":
    df = pd.read_csv("data/Personal_Finance_Dataset.csv")
    df = clean_transactions(df)
    monthly_df = create_monthly_features(df)

    #eda first
    run_eda(monthly_df)

    #then run model
    results_df, comparison_df = run_models(monthly_df)
    print("\nModel Results")
    print(results_df)

    print("\nPredictions Preview")
    print(comparison_df.head())

    plot_actual_vs_predicted(comparison_df, "Predicted_Random Forest")

