# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from utils.logger import setup_logger

# logger = setup_logger("DataPreprocessing")


# def load_data(path: str) -> pd.DataFrame:
#     logger.info("Loading dataset")
#     df = pd.read_csv(path)
#     return df

# def preprocess_data(df: pd.DataFrame):
#     logger.info("Starting preprocessing")

#     df = df.copy()

#     df.columns = df.columns.str.strip().str.lower()

#     df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
#     df = df.dropna(subset=["date"])

#     df.ffill(inplace=True)


#     le_category = LabelEncoder()
#     df["product_category_encoded"] = le_category.fit_transform(df["product_category"])

#     le_segment = LabelEncoder()
#     df["customer_segment_encoded"] = le_segment.fit_transform(df["customer_segment"])

#     # Winsorization
#     q_low = df["units_sold"].quantile(0.01)
#     q_high = df["units_sold"].quantile(0.99)
#     df["units_sold"] = df["units_sold"].clip(q_low, q_high)

#     prophet_df = (
#         df.groupby("date")
#         .agg({
#             "units_sold": "sum",
#             "price": "mean",
#             "discount": "mean",
#             "marketing_spend": "sum"
#         })
#         .reset_index()
#         .rename(columns={"date": "ds", "units_sold": "y"})
#     )

#     return df, prophet_df, {
#         "category_encoder": le_category,
#         "segment_encoder": le_segment
#     }

# if __name__ == "__main__":
#     df = load_data("data/commerce_Sales_Prediction_Dataset.csv")
#     processed_df, prophet_df, _ = preprocess_data(df)
#     print(prophet_df.head())

# preprocessing/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.logger import setup_logger

logger = setup_logger("DataPreprocessing")


def load_data(path: str) -> pd.DataFrame:
    logger.info("Loading dataset")
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame):
    """
    Cleans data, prevents leakage, and prepares Prophet-ready dataset.
    """

    logger.info("Starting preprocessing")

    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    # ---- Date handling ----
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")

    # ---- Group-wise forward fill (NO LEAKAGE) ----
    df = df.sort_values("date")

    df[["units_sold", "price", "discount", "marketing_spend"]] = (
        df.groupby(
            ["product_category", "customer_segment"],
            observed=True
        )[["units_sold", "price", "discount", "marketing_spend"]]
        .ffill()
    )


    # ---- Encoding (kept for ML extensibility) ----
    encoders = {}

    for col in ["product_category", "customer_segment"]:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col])
        encoders[col] = le

    # ---- Winsorization (robust outlier handling) ----
    for col in ["units_sold", "price", "discount", "marketing_spend"]:
        q_low, q_high = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(q_low, q_high)

    # ---- Prophet-ready aggregation ----
    prophet_df = (
        df.groupby("date", as_index=False)
        .agg(
            y=("units_sold", "sum"),
            price=("price", "mean"),
            discount=("discount", "mean"),
            marketing_spend=("marketing_spend", "sum"),
        )
        .rename(columns={"date": "ds"})
        .sort_values("ds")
    )

    logger.info("Preprocessing complete")

    return df, prophet_df, encoders
