import joblib
import numpy as np

def load_model(filepath):
    """
    Load the trained model from disk.

    Parameters:
    filepath (str): Path to the model file

    Returns:
    Trained model
    """
    try:
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file {filepath} not found. Please train the model first.")
        return None

def predict_water_requirement(model, input_features, dataset_path='../../Crop_recommendation.csv'):
    """
    Predict irrigation water requirement and find the most suitable crop for given features.

    Parameters:
    model: Trained model
    input_features (dict): Dictionary of feature values
    dataset_path (str): Path to the original dataset

    Returns:
    tuple: (predicted_water, recommended_crop, similarity_score)
    """
    # Convert input to DataFrame to match training format
    import pandas as pd
    features_df = pd.DataFrame([input_features])

    # Make water prediction
    prediction = model.predict(features_df)[0]

    # Load original dataset to find similar crops
    try:
        original_data = pd.read_csv(dataset_path)
        
        # Calculate similarity based on feature differences
        feature_cols = ['temperature', 'humidity', 'rainfall', 'N', 'P', 'K']
        similarities = []
        
        for _, row in original_data.iterrows():
            # Calculate Euclidean distance for the features
            distance = 0
            for col in feature_cols:
                distance += (input_features[col] - row[col]) ** 2
            similarity = 1 / (1 + distance)  # Convert distance to similarity score
            similarities.append(similarity)
        
        # Find the most similar crop
        best_idx = similarities.index(max(similarities))
        recommended_crop = original_data.iloc[best_idx]['label']
        similarity_score = max(similarities)
        
    except FileNotFoundError:
        recommended_crop = "Unknown"
        similarity_score = 0.0

    return prediction, recommended_crop, similarity_score

def main():
    """
    Main function to demonstrate prediction.
    """
    # Load model
    model_path = '../models/irrigation_model.pkl'
    model = load_model(model_path)

    if model is None:
        return

    # Example input features
    sample_input = {
        'temperature': 25.0,  # Celsius
        'humidity': 60.0,     # Percentage
        'rainfall': 50.0,     # mm
        'N': 80.0,            # Nitrogen
        'P': 40.0,            # Phosphorus
        'K': 40.0             # Potassium
    }

    # Make prediction
    prediction, recommended_crop, similarity = predict_water_requirement(model, sample_input)

    print("Input Features:")
    for key, value in sample_input.items():
        print(f"  {key}: {value}")

    print(f"Predicted water requirement: {prediction:.2f} units")
    print(f"Recommended crop: {recommended_crop}")
    print(".3f")

    # Additional examples
    print("\nAdditional Prediction Examples:")
    examples = [
        {'temperature': 30.0, 'humidity': 50.0, 'rainfall': 20.0, 'N': 90.0, 'P': 50.0, 'K': 50.0},
        {'temperature': 20.0, 'humidity': 80.0, 'rainfall': 100.0, 'N': 60.0, 'P': 30.0, 'K': 30.0},
        {'temperature': 28.0, 'humidity': 55.0, 'rainfall': 0.0, 'N': 100.0, 'P': 60.0, 'K': 60.0}
    ]

    for i, example in enumerate(examples, 1):
        pred, crop, sim = predict_water_requirement(model, example)
        print(f"Example {i}: Predicted water requirement: {pred:.2f} units, Recommended crop: {crop}")

if __name__ == "__main__":
    main()