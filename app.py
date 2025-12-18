import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction App")

st.write("Enter house details to predict the price:")

# Load dataset and train model (lightweight)
@st.cache_resource
def load_model():
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        random_state=42
    )
    model.fit(X, y)
    return model

model = load_model()

# User inputs
MedInc = st.number_input("Median Income", min_value=0.0)
HouseAge = st.number_input("House Age", min_value=0.0)
AveRooms = st.number_input("Average Rooms", min_value=0.0)
AveBedrms = st.number_input("Average Bedrooms", min_value=0.0)
Population = st.number_input("Population", min_value=0.0)
AveOccup = st.number_input("Average Occupancy", min_value=0.0)
Latitude = st.number_input("Latitude", min_value=0.0)
Longitude = st.number_input("Longitude", min_value=0.0)

if st.button("Predict Price"):
    input_data = pd.DataFrame([[MedInc, HouseAge, AveRooms, AveBedrms,
                                Population, AveOccup, Latitude, Longitude]],
                              columns=["MedInc", "HouseAge", "AveRooms",
                                       "AveBedrms", "Population", "AveOccup",
                                       "Latitude", "Longitude"])

    prediction = model.predict(input_data)
    st.success(f"💰 Predicted House Price: ${prediction[0]*100000:.2f}")
