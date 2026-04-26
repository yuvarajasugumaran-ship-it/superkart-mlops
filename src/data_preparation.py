import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import upload_file


HF_USERNAME = "Yuvisukumar"
HF_DATASET_REPO = f"{HF_USERNAME}/superkart-sales-dataset"

RAW_DATA_URL = (
    f"https://huggingface.co/datasets/{HF_DATASET_REPO}"
    "/resolve/main/SuperKart.csv"
)

TARGET_COLUMN = "Product_Store_Sales_Total"


def load_data():
    print("Loading raw dataset from Hugging Face...")
    df = pd.read_csv(RAW_DATA_URL)
    print("Raw dataset loaded successfully.")
    print(f"Raw dataset shape: {df.shape}")
    return df


def clean_data(df):
    print("Cleaning dataset...")

    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows where target column is missing
    df = df.dropna(subset=[TARGET_COLUMN])

    # Product_Id is mostly an identifier and not useful for general prediction
    df = df.drop(columns=["Product_Id"], errors="ignore")

    print("Data cleaning completed.")
    print(f"Cleaned dataset shape: {df.shape}")

    return df


def split_and_save_data(df):
    print("Splitting data into train and test datasets...")

    os.makedirs("data/processed", exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print("Train and test files saved locally.")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")


def upload_processed_files():
    print("Uploading train.csv and test.csv to Hugging Face Dataset Hub...")

    upload_file(
        path_or_fileobj="data/processed/train.csv",
        path_in_repo="train.csv",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset"
    )

    upload_file(
        path_or_fileobj="data/processed/test.csv",
        path_in_repo="test.csv",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset"
    )

    print("Upload completed successfully.")


if __name__ == "__main__":
    raw_df = load_data()
    cleaned_df = clean_data(raw_df)
    split_and_save_data(cleaned_df)
    upload_processed_files()