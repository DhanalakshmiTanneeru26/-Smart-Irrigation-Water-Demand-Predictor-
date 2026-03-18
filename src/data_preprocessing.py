import pandas as pd
import numpy as np
import os

def preprocess_data(data):
    """
    Perform basic preprocessing on the dataset.

    Parameters:
    data (pd.DataFrame): Input dataset

    Returns:
    pd.DataFrame: Preprocessed dataset
    """
    # Check for missing values
    print("Missing values:")
    print(data.isnull().sum())

    # Basic statistics
    print("\nDataset statistics:")
    print(data.describe())

    # Handle any missing values if present (though synthetic data shouldn't have any)
    data = data.dropna()

    return data

def main():
    """
    Main function to load and validate the Crop_recommendation dataset.
    """
    # Load the Crop_recommendation dataset
    crop_data_path = '../../Crop_recommendation.csv'
    try:
        data = pd.read_csv(crop_data_path)
        print(f"Loaded Crop_recommendation.csv with {data.shape[0]} rows and {data.shape[1]} columns")
    except FileNotFoundError:
        print(f"Error: {crop_data_path} not found.")
        return

    # Add water_required column for validation
    data['water_required'] = (
        data['temperature'] * 0.5 +
        (100 - data['humidity']) * 0.3 -
        data['rainfall'] * 0.01 +
        (data['N'] + data['P'] + data['K']) * 0.001 +
        np.random.normal(0, 5, len(data))  # Add some noise
    )

    # Ensure water_required is positive
    data['water_required'] = np.maximum(data['water_required'], 0)

    # Preprocess data
    print("Preprocessing data...")
    data = preprocess_data(data)

    print(f"Dataset ready for training. Shape: {data.shape}")
    print("Note: Dataset is processed in memory. No file saved.")

if __name__ == "__main__":
    main()