import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from huggingface_hub import upload_file


HF_USERNAME = "Yuvisukumar"

HF_DATASET_REPO = f"{HF_USERNAME}/superkart-sales-dataset"
HF_MODEL_REPO = f"{HF_USERNAME}/superkart-sales-model"

TRAIN_URL = (
    f"https://huggingface.co/datasets/{HF_DATASET_REPO}"
    "/resolve/main/train.csv"
)

TEST_URL = (
    f"https://huggingface.co/datasets/{HF_DATASET_REPO}"
    "/resolve/main/test.csv"
)

TARGET_COLUMN = "Product_Store_Sales_Total"


def load_train_test_data():
    print("Loading train and test datasets from Hugging Face...")

    train_df = pd.read_csv(TRAIN_URL)
    test_df = pd.read_csv(TEST_URL)

    print("Train and test datasets loaded successfully.")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    return train_df, test_df


def build_pipeline():
    numeric_features = [
        "Product_Weight",
        "Product_Allocated_Area",
        "Product_MRP",
        "Store_Establishment_Year"
    ]

    categorical_features = [
        "Product_Sugar_Content",
        "Product_Type",
        "Store_Id",
        "Store_Size",
        "Store_Location_City_Type",
        "Store_Type"
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features)
        ]
    )

    model = BaggingRegressor(
        estimator=DecisionTreeRegressor(random_state=42),
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


def train_and_tune_model(X_train, y_train):
    print("Training and tuning model...")

    pipeline = build_pipeline()

    param_grid = {
        "model__n_estimators": [50, 100],
        "model__max_samples": [0.8, 1.0],
        "model__max_features": [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("Model training completed.")
    print("Best parameters:")
    print(grid_search.best_params_)

    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    mape = (abs((y_test - predictions) / y_test).mean()) * 100

    print("Model Evaluation Results")
    print("------------------------")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R2   : {r2:.4f}")

    results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": predictions
    })

    results.to_csv("data/processed/test_predictions.csv", index=False)

    return mae, rmse, mape, r2

def save_and_upload_model(model):
    print("Saving best model locally...")

    model_path = "best_pipeline.joblib"
    joblib.dump(model, model_path)

    print("Uploading best model to Hugging Face Model Hub...")

    upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_pipeline.joblib",
        repo_id=HF_MODEL_REPO,
        repo_type="model"
    )

    print("Best model uploaded successfully.")


if __name__ == "__main__":
    train_df, test_df = load_train_test_data()

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    best_model = train_and_tune_model(X_train, y_train)

    evaluate_model(best_model, X_test, y_test)

    save_and_upload_model(best_model)