import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
# Set page configuration for a better look and feel
st.set_page_config(
    page_title="ML Model Deployment App",
    page_icon="", # Removed emoji
    layout="centered", # Can be "wide" or "centered"
    initial_sidebar_state="expanded" # Can be "auto", "expanded", or "collapsed"
)

# --- Load Model and Feature Names ---
@st.cache_resource # Cache the model loading to improve performance
def load_model_and_features():
    """Loads the pre-trained model and feature names."""
    try:
        model = joblib.load('logistic_regression_model.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, feature_names
    except FileNotFoundError:
        st.error("Error: Model files not found! Please run 'model_training.py' first.")
        st.stop() # Stop the app if model files are missing

model, feature_names = load_model_and_features()

# --- Title and Description ---
st.title("Simple ML Model Deployment App") # Removed emoji
st.markdown("""
Welcome to this interactive web application!
This app demonstrates how to deploy a machine learning model using Streamlit.
Input the features below and get a prediction from our pre-trained Logistic Regression model
(trained on the Iris dataset).
""")

# --- User Input Section ---
st.header(" Input Features")
st.markdown("Adjust the sliders to provide input for the model:")

# Create input widgets for each feature
input_data = {}
for feature in feature_names:
    # Use st.slider for numerical input, providing min/max/default values
    # These ranges are based on the Iris dataset's typical values
    if "sepal length" in feature:
        input_data[feature] = st.slider(f"**{feature.replace('_', ' ').title()} (cm)**", 4.0, 8.0, 5.8, 0.1)
    elif "sepal width" in feature:
        input_data[feature] = st.slider(f"**{feature.replace('_', ' ').title()} (cm)**", 2.0, 4.5, 3.0, 0.1)
    elif "petal length" in feature:
        input_data[feature] = st.slider(f"**{feature.replace('_', ' ').title()} (cm)**", 1.0, 7.0, 4.3, 0.1)
    elif "petal width" in feature:
        input_data[feature] = st.slider(f"**{feature.replace('_', ' ').title()} (cm)**", 0.1, 2.5, 1.3, 0.1)

# Convert input data to a DataFrame, ensuring column order matches training data
input_df = pd.DataFrame([input_data])
st.subheader("Your Input Data:")
st.dataframe(input_df)

# --- Prediction Button and Output ---
if st.button("Get Prediction"): # Removed emoji
    st.subheader("Prediction Results:")
    try:
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        # Map numerical prediction to class names (for Iris dataset)
        class_names = ['Setosa', 'Versicolor', 'Virginica']
        predicted_class_name = class_names[prediction]

        st.success(f"The model predicts: **{predicted_class_name}**")

        # --- Visualization of Probabilities ---
        st.subheader("Model Confidence (Probabilities):")
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(class_names, prediction_proba, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax.set_ylabel("Probability")
        ax.set_title("Probability of Each Class")
        ax.set_ylim(0, 1) # Ensure y-axis goes from 0 to 1

        # Add probability values on top of bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 3), ha='center', va='bottom')

        st.pyplot(fig)
        plt.close(fig) # Close the figure to prevent memory leaks

        st.markdown(f"""
        <div style="background-color:#e6ffe6; padding:10px; border-radius:5px; margin-top:15px;">
            The model is most confident that this input belongs to the **{predicted_class_name}** class.
            The probabilities show how likely the input is to be each of the possible classes.
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please ensure your input values are valid.")

# --- Footer ---
st.markdown("---")
st.markdown("Developedusing Streamlit and scikit-learn.")
st.markdown("This app is for demonstration purposes only.")
