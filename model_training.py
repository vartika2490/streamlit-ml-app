#This script is responsible for training a simple machine learning model and saving it to a file. In a real-world scenario, you'd likely have a more complex training pipeline, but for this assignment, it demonstrates how a model would be prepared for deployment. We'll use a basic Iris dataset and a Logistic Regression model for simplicity.
 #model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import joblib # Used for saving and loading models

def train_and_save_model():
    """
    Trains a Logistic Regression model on the Iris dataset
    and saves the trained model and feature names.
    """
    print("Starting model training...")

    # Load the Iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Logistic Regression model
    model = LogisticRegression(max_iter=200, solver='liblinear') # Increased max_iter for convergence
    model.fit(X_train, y_train)

    # Evaluate the model (optional, but good practice)
    accuracy = model.score(X_test, y_test)
    print(f"Model trained successfully! Accuracy on test set: {accuracy:.2f}")

    # Save the trained model
    model_filename = 'logistic_regression_model.pkl'
    joblib.dump(model, model_filename)
    print(f"Model saved as '{model_filename}'")

    # Save feature names (important for consistent input to the deployed model)
    feature_names_filename = 'feature_names.pkl'
    joblib.dump(iris.feature_names, feature_names_filename)
    print(f"Feature names saved as '{feature_names_filename}'")

if __name__ == "__main__":
    train_and_save_model()
