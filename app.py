import torch
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the scalers
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# Load the models
linear_model = torch.load('linear_regression_model.pth')
ann_model = torch.load('ann_model.pth')

# Streamlit app
st.title("Football Player Market Value Prediction")

# Model selection
model_choice = st.radio("Select Model:", ["Linear Regression", "ANN"])

# Feature input
feature_names = [
    "Age", "Dribbling / Reflexes", "Passing / Kicking", "Shooting / Handling",
    "International reputation", "Total mentality", "Shot power",
    "Total power", "Ball control", "Finishing"
]

player_features = []
for feature in feature_names:
    value = st.number_input(f"Enter value for {feature}:", value=0.0)
    player_features.append(value)

if st.button("Predict Market Value"):
    # Convert input to NumPy array and scale
    player_features = np.array(player_features).reshape(1, -1)
    player_features_scaled = scaler_X.transform(player_features)

    # Convert to PyTorch tensor
    player_features_tensor = torch.tensor(player_features_scaled, dtype=torch.float32)

    if model_choice == "Linear Regression":
        # Linear regression prediction
        w = linear_model['weights']
        b = linear_model['bias']
        log_market_value = player_features_tensor @ w + b

    elif model_choice == "ANN":
        # ANN prediction
        w1 = ann_model['w1']
        b1 = ann_model['b1']
        w2 = ann_model['w2']
        b2 = ann_model['b2']
        w3 = ann_model['w3']
        b3 = ann_model['b3']

        # Forward pass
        h1 = player_features_tensor @ w1 + b1
        a1 = torch.relu(h1)
        h2 = a1 @ w2 + b2
        a2 = torch.relu(h2)
        log_market_value = a2 @ w3 + b3

    # Convert log-scale output back to original scale
    log_market_value_numpy = log_market_value.detach().numpy().reshape(-1, 1)
    predicted_value_scaled = scaler_y.inverse_transform(log_market_value_numpy)
    predicted_value = np.exp(predicted_value_scaled).item()  # Extract scalar value

    st.write(f"Predicted Market Value: {predicted_value:.2f}")
