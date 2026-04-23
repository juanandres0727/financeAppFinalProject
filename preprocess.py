import pandas as pd

"""
    we handle data preparation here 
"""

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    #standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    #convert date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    #remove rows with invalid dates or amount
    df = df.dropna(subset=["date", "amount"])

    #standardize text columns
    df["category"] = df["category"].astype(str).str.strip().str.title()
    df["type"] = df["type"].astype(str).str.strip().str.title()
    df["transaction_description"] = df["transaction_description"].astype(str).str.strip()

    #remove duplicates
    df = df.drop_duplicates()

    #keep only expenses for spending prediction
    df = df[df["type"] == "Expense"]

    #remove impossible negative expenses if needed
    df = df[df["amount"] >= 0]

    return df


def create_monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year_month"] = df["date"].dt.to_period("M")

    monthly_total = (
        df.groupby("year_month")["amount"]
        .sum()
        .reset_index(name="monthly_spending")
    )

    monthly_tx_count = (
        df.groupby("year_month")
        .size()
        .reset_index(name="transaction_count")
    )

    category_pivot = (
        df.pivot_table(
            index="year_month",
            columns="category",
            values="amount",
            aggfunc="sum",
            fill_value=0
        )
        .reset_index()
    )

    monthly_df = monthly_total.merge(monthly_tx_count, on="year_month", how="left")
    monthly_df = monthly_df.merge(category_pivot, on="year_month", how="left")

    monthly_df = monthly_df.sort_values("year_month").reset_index(drop=True)

    monthly_df["month"] = monthly_df["year_month"].dt.month
    monthly_df["quarter"] = monthly_df["year_month"].dt.quarter
    monthly_df["year"] = monthly_df["year_month"].dt.year

    monthly_df["lag_1"] = monthly_df["monthly_spending"].shift(1)
    monthly_df["lag_2"] = monthly_df["monthly_spending"].shift(2)
    monthly_df["lag_3"] = monthly_df["monthly_spending"].shift(3)

    monthly_df["rolling_mean_3"] = monthly_df["monthly_spending"].shift(1).rolling(3).mean()
    monthly_df["rolling_std_3"] = monthly_df["monthly_spending"].shift(1).rolling(3).std()

    monthly_df["spending_change_1"] = (
        monthly_df["monthly_spending"].shift(1) - monthly_df["monthly_spending"].shift(2)
    )

    monthly_df["target_next_month_spending"] = monthly_df["monthly_spending"].shift(-1)

    monthly_df["year_month"] = monthly_df["year_month"].astype(str)

    return monthly_df.dropna().reset_index(drop=True)