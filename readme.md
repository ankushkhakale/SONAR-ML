SONAR Rock vs. Mine Prediction

By Ankush Khakale

📖 Project Overview
This project uses a Logistic Regression machine learning model to predict whether an object detected by SONAR is a Rock or a Mine. The model is trained on the "Sonar, Mines vs. Rocks" dataset, which contains 60 sensor readings for 208 different objects.

The Python script (ML2.py) handles the entire machine learning workflow:

Data Loading: Loads the dataset from a local CSV file.

Data Analysis: Provides a basic statistical summary of the data.

Data Preparation: Splits the data into features (X) and labels (Y) and encodes the labels.

Model Training: Trains a Logistic Regression classifier on 80% of the data.

Model Evaluation: Measures the model's accuracy on both the training and testing data.

Predictive System: Allows a user to input 60 new sonar readings to get a real-time prediction.