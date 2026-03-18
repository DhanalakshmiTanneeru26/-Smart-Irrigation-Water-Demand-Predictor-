# Smart Irrigation Water Demand Predictor

A machine learning regression system that predicts the amount of irrigation water required for crops based on environmental and soil conditions, and recommends suitable crops for the given conditions.

## Project Overview

This project builds a RandomForestRegressor model to predict irrigation water requirements using features like temperature, humidity, rainfall, and soil nutrients (N, P, K).

## Dataset

The project uses the `Crop_recommendation.csv` dataset (located in the parent directory) with the following features:
- temperature (°C)
- humidity (%)
- rainfall (mm)
- N (nitrogen)
- P (phosphorus)
- K (potassium)

The target variable **water_required** is computed based on environmental conditions and added dynamically during data loading.

## Features

- Data preprocessing and exploration
- Exploratory Data Analysis (EDA) with visualizations
- Machine learning model training and evaluation
- Model serialization with joblib
- Prediction script for new data with crop recommendations
- Flask web application for user interface with complete predictions

## Installation

1. Clone or download the project.
2. Navigate to the project directory.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing

Run the data preprocessing script to validate the dataset:

```bash
python src/data_preprocessing.py
```

This loads the Crop_recommendation.csv and validates the data (no intermediate file is created).

### 2. Exploratory Data Analysis

Open and run the Jupyter notebook:

```bash
jupyter notebook notebooks/analysis.ipynb
```

### 3. Train the Model

Train the machine learning model:

```bash
python src/train_model.py
```

This saves the trained model to `models/irrigation_model.pkl`.

### 4. Make Predictions

Use the prediction script:

```bash
python src/predict.py
```

Modify the input values in the script to get predictions.

### 5. Run the Web App

Start the Flask application:

```bash
python app/app.py
```

Open your browser to `http://127.0.0.1:5000/` to access the web interface.

**Web Interface Features:**
- **Home Page (/)**: Input form for environmental and soil parameters
- **Results Page (/predict)**: Displays water requirement prediction and crop recommendation
- Clean, responsive design with form validation
- Real-time predictions using the trained RandomForest model

## Project Structure

```
smart-irrigation-ml-project/
│
├── notebooks/
│   └── analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── predict.py
│
├── models/
│   └── irrigation_model.pkl
│
├── app/
│   └── app.py
│
├── requirements.txt
│
└── README.md
```

Note: The project loads `Crop_recommendation.csv` directly from the parent directory and processes it in memory.

## Model Evaluation

The model is evaluated using:
- R² Score
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

Additionally, the system provides crop recommendations by finding the most similar crop from the dataset based on environmental conditions.

## Technologies Used

- Python
- Pandas & NumPy for data manipulation
- Matplotlib & Seaborn for visualization
- Scikit-learn for machine learning
- Joblib for model serialization
- Flask for web application