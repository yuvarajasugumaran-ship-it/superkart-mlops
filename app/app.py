import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download


HF_USERNAME = "Yuvisukumar"
HF_MODEL_REPO = f"{HF_USERNAME}/superkart-sales-model"


st.set_page_config(
    page_title="SuperKart Sales Forecast App",
    page_icon="📈",
    layout="centered"
)


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename="best_pipeline.joblib"
    )
    model = joblib.load(model_path)
    return model


model = load_model()


st.title("SuperKart Sales Forecast App")

st.write(
    "Enter product and store details below to predict product-store sales revenue."
)


product_weight = st.number_input(
    "Product Weight",
    min_value=0.0,
    value=12.0
)

product_sugar_content = st.selectbox(
    "Product Sugar Content",
    ["Low Sugar", "Regular", "No Sugar"]
)

product_allocated_area = st.number_input(
    "Product Allocated Area",
    min_value=0.0,
    max_value=1.0,
    value=0.05
)

product_type = st.selectbox(
    "Product Type",
    [
        "Baking Goods",
        "Breads",
        "Breakfast",
        "Canned",
        "Dairy",
        "Frozen Foods",
        "Fruits and Vegetables",
        "Hard Drinks",
        "Health and Hygiene",
        "Household",
        "Meat",
        "Others",
        "Seafood",
        "Snack Foods",
        "Soft Drinks",
        "Starchy Foods"
    ]
)

product_mrp = st.number_input(
    "Product MRP",
    min_value=0.0,
    value=120.0
)

store_id = st.selectbox(
    "Store ID",
    [
        "OUT001", "OUT002", "OUT003", "OUT004", "OUT005",
        "OUT006", "OUT007", "OUT008", "OUT009", "OUT010"
    ]
)

store_establishment_year = st.number_input(
    "Store Establishment Year",
    min_value=1900,
    max_value=2026,
    value=2009
)

store_size = st.selectbox(
    "Store Size",
    ["Small", "Medium", "High"]
)

store_location_city_type = st.selectbox(
    "Store Location City Type",
    ["Tier 1", "Tier 2", "Tier 3"]
)

store_type = st.selectbox(
    "Store Type",
    [
        "Departmental Store",
        "Food Mart",
        "Supermarket Type1",
        "Supermarket Type2",
        "Supermarket Type3"
    ]
)


input_df = pd.DataFrame([{
    "Product_Weight": product_weight,
    "Product_Sugar_Content": product_sugar_content,
    "Product_Allocated_Area": product_allocated_area,
    "Product_Type": product_type,
    "Product_MRP": product_mrp,
    "Store_Id": store_id,
    "Store_Establishment_Year": store_establishment_year,
    "Store_Size": store_size,
    "Store_Location_City_Type": store_location_city_type,
    "Store_Type": store_type
}])


if st.button("Predict Sales"):
    prediction = model.predict(input_df)[0]

    st.success(
        f"Predicted Product Store Sales Total: {prediction:,.2f}"
    )

    st.subheader("Input Data Used for Prediction")
    st.dataframe(input_df)