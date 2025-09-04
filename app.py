# -----------------------------------
# Streamlit App for Model Deployment
# -----------------------------------
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time

# --- IMPORTANT NOTE ON SCALER ---
# In a real-world deployment, you must save and load the exact same StandardScaler
# object that was fitted on your training data. This ensures that new input data
# is scaled correctly before being fed into the model. For this demonstration,
# a mock scaler is used, but it's crucial to understand this is for illustration
# purposes only and should be replaced with a proper saved scaler in production.

class MockScaler(StandardScaler):
    """
    A mock scaler class to simulate loading a pre-trained scaler.
    The mean and scale values are hardcoded from the original training script.
    """
    def __init__(self):
        super().__init__()
        # These values would come from saving the fitted scaler object
        self.mean_ = np.array([94819.5, 88.35])
        self.scale_ = np.array([47481.5, 250.21])

    def transform(self, X):
        # This method assumes X has two columns: Time and Amount
        return (X - self.mean_) / self.scale_

# --- Model and Scaler Loading ---
# This function is decorated with Streamlit's cache decorator, which means it will
# only run once when the app is first launched, saving time on subsequent runs.
@st.cache_resource
def load_model_and_scaler():
    """Loads the trained model and returns a mock scaler."""
    try:
        model = tf.keras.models.load_model('credit_card_fraud_model.keras')
        scaler = MockScaler()
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.stop()

model, scaler = load_model_and_scaler()

# -----------------------------------
# Streamlit UI
# -----------------------------------
st.title('Secure Transaction Hub')
st.subheader('Credit Card Fraud Detection Service')

st.markdown("""
Welcome to the Secure Transaction Hub. Please provide the transaction details
below to check for potential fraudulent activity. All sensitive data is anonymized
for your privacy.
""")

# Use columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    time_val = st.number_input(
        'Time (in seconds from first transaction)', 
        min_value=0.0, 
        value=0.0,
        help="The time elapsed between this transaction and the first transaction in the dataset."
    )

with col2:
    amount_val = st.number_input(
        'Transaction Amount', 
        min_value=0.0, 
        value=0.0, 
        format="%.2f",
        help="The monetary value of the transaction."
    )

# Use an expander to hide the complex PCA features
with st.expander("Enter Anonymized Transaction Features (V1-V28)"):
    st.info("The features V1 through V28 are the principal components obtained with PCA. The exact meaning of these features is confidential to protect user privacy.")
    v_inputs = {}
    for i in range(1, 29):
        v_inputs[f'V{i}'] = st.number_input(f'V{i}', value=0.0)

# Prediction button in the main area
if st.button('Analyze Transaction', use_container_width=True, type="primary"):
    with st.spinner('Analyzing transaction...'):
        time.sleep(2)  # Simulate a brief delay for a better user experience

        # Create a dictionary of all inputs
        transaction_data = {**v_inputs, 'Time': time_val, 'Amount': amount_val}
        
        # Order the features correctly as per the original dataset
        # This is a crucial step for the model to work correctly
        feature_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        input_list = [transaction_data[f] for f in feature_order]
        
        # Convert input to a numpy array with the correct shape
        input_array = np.array(input_list).reshape(1, -1)
        
        # Preprocess the input data
        # Only scale the Time and Amount features as done in the original training script
        input_array[:, [0, -1]] = scaler.transform(input_array[:, [0, -1]])
        
        # Make prediction
        prediction_prob = model.predict(input_array)[0][0]
        
        st.markdown('---')
        st.subheader('Prediction Result')
        
        # Display a clear result with a colored box
        if prediction_prob > 0.5:
            st.error(f'❌ **Prediction: Fraudulent Transaction**')
            st.markdown(f'**Confidence**: `{prediction_prob * 100:.2f}%`')
        else:
            st.success(f'✅ **Prediction: Legitimate Transaction**')
            st.markdown(f'**Confidence**: `{(1 - prediction_prob) * 100:.2f}%`')

st.markdown("""
<style>
.st-emotion-cache-1r6504v {
    background-color: #f0f0f0;
}
</style>
""", unsafe_allow_html=True)
