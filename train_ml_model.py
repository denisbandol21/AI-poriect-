import os
import pandas as pd
import opendatasets as od
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_ml_model():
    print("--- Starting ML Model Training ---")

    dataset_url = 'https://www.kaggle.com/datasets/anikannal/solar-power-generation-data'
    data_dir = 'data'
    if not os.path.exists(os.path.join(data_dir, 'solar-power-generation-data')):
        print("Downloading dataset from Kaggle...")
        print("Please enter your Kaggle credentials when prompted.")
        od.download(dataset_url, data_dir=data_dir)
    else:
        print("Dataset already exists. Skipping download.")

    print("Loading and preprocessing data...")
    try:
        generation_data_path = os.path.join(data_dir, 'solar-power-generation-data', 'Plant_1_Generation_Data.csv')
        weather_data_path = os.path.join(data_dir, 'solar-power-generation-data', 'Plant_1_Weather_Sensor_Data.csv')
        generation_data = pd.read_csv(generation_data_path)
        weather_data = pd.read_csv(weather_data_path)
    except FileNotFoundError:
        print("\nERROR: CSV files not found after download.")
        print("Please check the 'data/solar-power-generation-data' directory.")
        return

    generation_data['DATE_TIME'] = pd.to_datetime(generation_data['DATE_TIME'], format='%d-%m-%Y %H:%M')
    weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

    merged_data = pd.merge(generation_data, weather_data, on='DATE_TIME')
    merged_data['hour'] = merged_data['DATE_TIME'].dt.hour
    merged_data['day_of_year'] = merged_data['DATE_TIME'].dt.dayofyear

    print("Data loaded and preprocessed successfully.")

    print("Training the Linear Regression model...")
    features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'hour', 'day_of_year']
    target = 'DC_POWER'

    merged_data.dropna(subset=features + [target], inplace=True)

    X = merged_data[features]
    y = merged_data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model training complete. Mean Squared Error: {mse:.2f}, R^2 Score: {r2:.2f}")

    output_dir = 'src/ml'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model_filename = os.path.join(output_dir, 'solar_power_prediction_model.joblib')
    joblib.dump(model, model_filename)
    print(f"Model saved successfully to '{model_filename}'")
    print("--- ML Model Training Finished ---")

if __name__ == "__main__":
    train_ml_model()
