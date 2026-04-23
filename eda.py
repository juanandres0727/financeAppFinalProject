import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
    we get the visual and descriptive analysis
"""

def plot_monthly_spending(monthly_df: pd.DataFrame):
    plt.figure()
    plt.plot(monthly_df["year_month"], monthly_df["monthly_spending"])
    plt.title("Monthly Spending Over Time")
    plt.xlabel("Month")
    plt.ylabel("Spending")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_category_totals(monthly_df: pd.DataFrame):
    category_cols = [
        "Entertainment",
        "Food & Drink",
        "Health & Fitness",
        "Rent",
        "Salary",
        "Shopping",
        "Travel",
        "Utilities"
    ]

    existing_cols = [col for col in category_cols if col in monthly_df.columns]

    category_totals = monthly_df[existing_cols].sum().sort_values(ascending=False)

    plt.figure()
    category_totals.plot(kind="bar")
    plt.title("Total Spending by Category")
    plt.ylabel("Amount")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(monthly_df: pd.DataFrame):
    numeric_df = monthly_df.select_dtypes(include=["number"])
    plt.figure()
    sns.heatmap(numeric_df.corr(), annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def run_eda(monthly_df: pd.DataFrame):
    print("\n=== EDA SUMMARY ===")
    print("Shape:", monthly_df.shape)
    print("\nColumns:")
    print(monthly_df.columns.tolist())
    print("\nFirst rows:")
    print(monthly_df.head())
    print("\nSummary stats:")
    print(monthly_df.describe())
    plot_monthly_spending(monthly_df)
    plot_category_totals(monthly_df)
    plot_correlation_heatmap(monthly_df)