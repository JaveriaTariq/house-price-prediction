import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction App")
st.write(
    "Use this app to predict house prices in California. "
    "Enter the house details below and click **Predict Price**."
)

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
    return model, X

model, X_data = load_model()

# Sidebar inputs using sliders for better UX
st.sidebar.header("House Features")
MedInc = st.sidebar.slider("Median Income (10k USD)", 0.0, 15.0, 3.0)
HouseAge = st.sidebar.slider("House Age (years)", 0, 50, 10)
AveRooms = st.sidebar.slider("Average Rooms", 1, 20, 5)
AveBedrms = st.sidebar.slider("Average Bedrooms", 1, 10, 2)
Population = st.sidebar.slider("Population", 1, 5000, 1000)
AveOccup = st.sidebar.slider("Average Occupancy", 1, 10, 3)
Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 37.0)
Longitude = st.sidebar.slider("Longitude", -124.0, -114.0, -119.0)

# Validation example: no negative numbers
inputs = [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
if any(x < 0 for x in inputs):
    st.error("All values must be non-negative!")
else:
    if st.button("Predict Price"):
        input_data = pd.DataFrame([[MedInc, HouseAge, AveRooms, AveBedrms,
                                    Population, AveOccup, Latitude, Longitude]],
                                  columns=X_data.columns)
        prediction = model.predict(input_data)
        st.success(f"💰 Predicted House Price: ${prediction[0]*100000:.2f}")

        # Model explanation using SHAP
        with st.expander("Explain Prediction (Feature Importance)"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_data)
            shap.initjs()
            st.write("Global Feature Importance:")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_data, show=False)
            st.pyplot(fig)
