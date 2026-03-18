import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

def load_data(filepath):
    """
    Load the Crop_recommendation dataset and add water_required column.

    Parameters:
    filepath (str): Path to the CSV file

    Returns:
    pd.DataFrame: Loaded and processed dataset
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {data.shape}")
        
        # Add water_required column based on environmental conditions
        data['water_required'] = (
            data['temperature'] * 0.5 +
            (100 - data['humidity']) * 0.3 -
            data['rainfall'] * 0.01 +
            (data['N'] + data['P'] + data['K']) * 0.001 +
            np.random.normal(0, 5, len(data))  # Add some noise
        )
        
        # Ensure water_required is positive
        data['water_required'] = np.maximum(data['water_required'], 0)
        
        print(f"Added water_required column. Final shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None

def preprocess_features(data):
    """
    Preprocess features for model training.

    Parameters:
    data (pd.DataFrame): Input dataset

    Returns:
    tuple: Features (X) and target (y)
    """
    # Features
    feature_cols = ['temperature', 'humidity', 'rainfall', 'N', 'P', 'K']
    X = data[feature_cols]

    # Target
    y = data['water_required']

    return X, y

def train_model(X_train, y_train):
    """
    Train the RandomForestRegressor model.

    Parameters:
    X_train (pd.DataFrame): Training features
    y_train (pd.Series): Training target

    Returns:
    RandomForestRegressor: Trained model
    """
    # Initialize model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )

    # Train model
    model.fit(X_train, y_train)
    print("Model trained successfully.")

    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.

    Parameters:
    model: Trained model
    X_test (pd.DataFrame): Test features
    y_test (pd.Series): Test target
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("Model Evaluation:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")

    return r2, mse, rmse

def save_model(model, filepath):
    """
    Save the trained model to disk.

    Parameters:
    model: Trained model
    filepath (str): Path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def main():
    """
    Main function to train and save the irrigation prediction model.
    """
    # Load data
    data_path = '../../Crop_recommendation.csv'
    data = load_data(data_path)

    if data is None:
        return

    # Preprocess features
    X, y = preprocess_features(data)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Save model
    model_path = '../models/irrigation_model.pkl'
    save_model(model, model_path)

if __name__ == "__main__":
    main()