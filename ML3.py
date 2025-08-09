# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# --- 1. Data Collection and Pre-processing ---

# Load the dataset from a CSV file into a pandas DataFrame
# The dataset is publicly available from the UCI Machine Learning Repository.
# For this script, we assume the 'Copy of sonar data.csv' file is in the same directory.
try:
    sonar_data = pd.read_csv(r'C:\Users\khaka\Downloads\Copy of sonar data.csv', header=None)
except FileNotFoundError:
    print("Error: 'Copy of sonar data.csv' not found.")
    print("Please make sure the file is in the same directory as the script.")
    exit()

# --- 2. Exploratory Data Analysis (EDA) ---

# Display the first 5 rows of the dataframe
print("--- First 5 Rows of the Dataset ---")
print(sonar_data.head())
print("\n" + "=" * 40 + "\n")

# Get the dimensions of the dataframe (rows, columns)
print(f"Dataset contains {sonar_data.shape[0]} rows and {sonar_data.shape[1]} columns.")
print("\n" + "=" * 40 + "\n")

# Get a statistical summary of the data
print("--- Statistical Summary of the Dataset ---")
print(sonar_data.describe())
print("\n" + "=" * 40 + "\n")

# Count the occurrences of 'R' (Rock) and 'M' (Mine)
print("--- Class Distribution ---")
print(sonar_data[60].value_counts())
print("\n" + "=" * 40 + "\n")

# --- 3. Data Preparation ---

# Separate the data (features) from the labels (target)
# Features are in columns 0-59, the label is in column 60
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

# Encode the categorical labels ('R' and 'M') into numerical data
# LabelEncoder sorts labels alphabetically: 'M' becomes 0, 'R' becomes 1.
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# --- 4. Train-Test Split ---

# Split the data into training and testing sets
# 80% of the data will be used for training, and 20% for testing.
# random_state ensures that the split is the same every time the code is run.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.2, stratify=Y_encoded, random_state=1)

print("--- Data Split ---")
print("Original data shape:", X.shape)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("\n" + "=" * 40 + "\n")

# --- 5. Model Training: Logistic Regression ---

# Create a Logistic Regression model instance
model = LogisticRegression()

# Train the model with the training data
model.fit(X_train, Y_train)

# --- 6. Model Evaluation ---

# Make predictions on the training data to check for overfitting
training_data_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(training_data_prediction, Y_train)
print(f"Accuracy on training data: {training_data_accuracy * 100:.2f}%")

# Make predictions on the test data to evaluate the model's performance
test_data_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(test_data_prediction, Y_test)
print(f"Accuracy on test data: {test_data_accuracy * 100:.2f}%")
print("\n" + "=" * 40 + "\n")

# --- 7. Building a Predictive System ---

print("--- Predictive System ---")
print("Enter 60 comma-separated sonar readings:")

try:
    # Get input from the user in the terminal
    input_str = input()

    # Split the string by commas and convert to a list of floats
    input_data = [float(val.strip()) for val in input_str.split(',')]

    # Check if the user provided exactly 60 features
    if len(input_data) != 60:
        print(f"Error: Expected 60 feature values, but got {len(input_data)}.")
    else:
        # Convert the input data to a numpy array
        input_data_as_numpy_array = np.asarray(input_data)

        # Reshape the numpy array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # Make a prediction
        prediction = model.predict(input_data_reshaped)

        # 'M' (Mine) is encoded as 0, 'R' (Rock) is encoded as 1.
        if prediction[0] == 1:
            print("Prediction: The object is a Rock (R)")
        else:
            print("Prediction: The object is a Mine (M)")

except ValueError:
    print("Error: Invalid input. Please enter only numbers separated by commas.")
except Exception as e:
    print(f"An error occurred: {e}")

print("\n" + "=" * 40 + "\n")
