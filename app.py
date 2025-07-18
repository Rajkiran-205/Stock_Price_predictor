import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load Models ---
# Use a try-except block to handle potential FileNotFoundError
try:
    model = joblib.load('xgb_stock_predictor.joblib')
    scaler = joblib.load('scaler.joblib')
    model_columns = joblib.load('model_columns.joblib')
except FileNotFoundError:
    st.error("Model files not found. Please ensure the .joblib files are in the same directory as the app.")
    st.stop()

# --- Streamlit App ---

st.title('Stock Price Predictor (XGBoost)')
st.write('This app predicts the closing price of a stock based on fundamental indicators.')

# --- User Input ---
st.sidebar.header('Input Features')

def user_input_features():
    # **CHANGE:** Replaced technical indicators with fundamental ones.
    total_revenue = st.sidebar.number_input('Total Revenue (in billions)', value=250.0, key='total_revenue')
    eps = st.sidebar.number_input('Earnings Per Share (EPS)', value=4.0, format="%.2f", key='eps')
    pe_ratio = st.sidebar.number_input('P/E Ratio', value=25.0, format="%.2f", key='pe_ratio')
    
    # Sector selection
    sectors = sorted([col.replace('Sector_', '') for col in model_columns if 'Sector_' in col])
    sector_choice = st.sidebar.selectbox('GICS Sector', sectors, key='sector')

    # Create a dictionary of the inputs
    data = {
        'Total Revenue': total_revenue * 1e9, # Convert back to billions
        'Earnings Per Share': eps,
        'P/E_Ratio': pe_ratio,
    }
    
    # Add one-hot encoded sector data
    for s in sectors:
        data[f'Sector_{s}'] = 1 if sector_choice == s else 0
        
    # Add placeholders for other features that the model expects
    # These are less critical for user input but needed for the model's feature vector
    placeholder_features = {
        'MA_50': 100.0,
        'MA_200': 95.0,
        'RSI_14': 55.0,
        'Market_Cap': (pe_ratio * eps) * 5e7, # Estimated market cap
    }
    data.update(placeholder_features)
    
    # Convert to dataframe
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Prediction ---
# The prediction logic is now inside a button to make it explicit
if st.sidebar.button('Predict'):
    # Create a full dataframe with all model columns, initialized to 0
    # This ensures the input matches the model's training columns exactly
    full_input_df = pd.DataFrame(columns=model_columns)
    # Use pd.concat and reindex to safely merge the user input
    full_input_df = pd.concat([full_input_df, input_df]).reindex(columns=model_columns).fillna(0)

    # Separate columns for scaling
    features_to_scale = [col for col in model_columns if 'Sector_' not in col]
    
    # Apply the scaler
    # Note: We use .transform(), not .fit_transform(), as the scaler is already fitted
    input_scaled = scaler.transform(full_input_df[features_to_scale])
    input_scaled_df = pd.DataFrame(input_scaled, columns=features_to_scale)

    # Combine scaled numerical features with the unscaled one-hot encoded features
    final_input = pd.concat([input_scaled_df, full_input_df.drop(columns=features_to_scale)], axis=1)

    # Make prediction
    prediction = model.predict(final_input)

    # --- Display Output ---
    st.subheader('Input Features:')
    st.write(input_df[['Total Revenue', 'Earnings Per Share', 'P/E_Ratio']]) # Display only the main user inputs

    st.subheader('Prediction:')
    st.write(f'The predicted closing price is: **${prediction[0]:.2f}**')
else:
    st.info('Adjust the features in the sidebar and click "Predict" to see the result.')