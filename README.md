# 🏠 House Price Prediction App (Regression)

Predict California house prices using a Random Forest regression model. The app allows users to input housing features and instantly get a predicted house price.

## 🚀 Live Demo
[Streamlit App Link](http://house-price-prediction-efv4sbux7jbts9swrtqt9w.streamlit.app/)

## 📚 Features
- Enter housing features using a **user-friendly interface**.
- Instant house price prediction.
- Lightweight model trained inside the app for fast deployment.
- Input validation to ensure correct data entry.

## 🧰 Tech Stack
- Python
- Streamlit
- Scikit-learn
- Pandas, NumPy

## 📝 ML Workflow
1. Load California Housing dataset.
2. Explore & visualize data (EDA).
3. Train a Random Forest Regressor (lightweight for deployment).
4. Predict house prices from user input.
5. Cache model for fast predictions.

## 📊 Model Explanation
- **Algorithm:** Random Forest Regressor  
- **Reason:** Captures non-linear relationships between housing features and prices.  
- **Key Features:**
  - Median Income (`MedInc`) – most important
  - Average Rooms (`AveRooms`)
  - House Age (`HouseAge`)  
- **Performance:** Lightweight model optimized for fast predictions in Streamlit.

## 🎨 UI Improvements
- Inputs as **sliders** and number fields for better UX.
- Validation: Inputs cannot be negative.
- Dynamic output formatting with currency display.

## 💡 How to Run Locally
1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/house-price-prediction.git
   cd house-price-prediction
