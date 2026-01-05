import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.logger import setup_logger

logger = setup_logger("DataPreprocessing")


def load_data(path: str) -> pd.DataFrame:
    logger.info("Loading dataset")
    df = pd.read_csv(path)
    return df


# def preprocess_data(df: pd.DataFrame):
#     logger.info("Starting preprocessing")

#     # Normalize column names
#     df.columns = df.columns.str.strip().str.lower()

#     # Convert Date column
#     df["date"] = pd.to_datetime(
#         df["date"],
#         dayfirst=True,
#         errors="coerce"
#     )
#     df = df.dropna(subset=["date"])

#     # 🎯 TARGET VARIABLE (VERY IMPORTANT)
#     sales_col = "units_sold"   # <--- THIS IS THE ONLY REQUIRED CHANGE

#     if sales_col not in df.columns:
#         raise KeyError("Column 'Units_Sold' not found in dataset")

#     # Fill missing values
#     df.fillna(method="ffill", inplace=True)

#     # Encode categorical columns
#     categorical_cols = [
#         "product_category",
#         "customer_segment"
#     ]

#     encoders = {}
#     for col in categorical_cols:
#         from sklearn.preprocessing import LabelEncoder
#         le = LabelEncoder()
#         df[col + "_encoded"] = le.fit_transform(df[col])
#         encoders[col] = le

#     # Prepare Prophet DataFrame
#     prophet_df = (
#         df.groupby("date")[sales_col]
#         .sum()
#         .reset_index()
#         .rename(columns={"date": "ds", sales_col: "y"})
#     )

#     logger.info("Preprocessing completed successfully")
#     return df, prophet_df, encoders

# def preprocess_data(df: pd.DataFrame):
#     logger.info("Starting preprocessing")

#     # Normalize column names
#     df.columns = df.columns.str.strip().str.lower()

#     # Date parsing
#     df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
#     df = df.dropna(subset=["date"])

#     # Target
#     sales_col = "units_sold"

#     # Fill missing values
#     df.fillna(method="ffill", inplace=True)

#     # Encode categories
#     from sklearn.preprocessing import LabelEncoder
#     le_category = LabelEncoder()
#     df["product_category_encoded"] = le_category.fit_transform(df["product_category"])

#     le_segment = LabelEncoder()
#     df["customer_segment_encoded"] = le_segment.fit_transform(df["customer_segment"])


#     # Remove extreme sales spikes (winsorization)
#     q_low = df["units_sold"].quantile(0.01)
#     q_high = df["units_sold"].quantile(0.99)
#     df["units_sold"] = df["units_sold"].clip(q_low, q_high)


#     # Prophet dataframe with regressors
#     prophet_df = (
#         df.groupby("date")
#         .agg({
#             sales_col: "sum",
#             "price": "mean",
#             "discount": "mean",
#             "marketing_spend": "sum"
#         })
#         .reset_index()
#         .rename(columns={
#             "date": "ds",
#             sales_col: "y"
#         })
#     )

#     return df, prophet_df, {
#         "category_encoder": le_category,
#         "segment_encoder": le_segment
#     }

def preprocess_data(df: pd.DataFrame):
    logger.info("Starting preprocessing")

    df = df.copy()

    df.columns = df.columns.str.strip().str.lower()

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])

    df.fillna(method="ffill", inplace=True)

    le_category = LabelEncoder()
    df["product_category_encoded"] = le_category.fit_transform(df["product_category"])

    le_segment = LabelEncoder()
    df["customer_segment_encoded"] = le_segment.fit_transform(df["customer_segment"])

    # Winsorization
    q_low = df["units_sold"].quantile(0.01)
    q_high = df["units_sold"].quantile(0.99)
    df["units_sold"] = df["units_sold"].clip(q_low, q_high)

    prophet_df = (
        df.groupby("date")
        .agg({
            "units_sold": "sum",
            "price": "mean",
            "discount": "mean",
            "marketing_spend": "sum"
        })
        .reset_index()
        .rename(columns={"date": "ds", "units_sold": "y"})
    )

    return df, prophet_df, {
        "category_encoder": le_category,
        "segment_encoder": le_segment
    }




if __name__ == "__main__":
    df = load_data("data/commerce_Sales_Prediction_Dataset.csv")
    processed_df, prophet_df, _ = preprocess_data(df)
    print(prophet_df.head())
