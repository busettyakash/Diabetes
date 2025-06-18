# Diabetes Prediction Web Application

This web application allows users to predict diabetes using various machine learning models. Users can upload CSV files containing patient data, and the application will provide predictions using the selected model.

## Features

- Upload CSV files with patient data
- Choose from multiple machine learning models:
  - Decision Tree (DT)
  - K-Nearest Neighbors (KNN)
  - Random Forest (RF)
  - Naive Bayes (NB)
  - AdaBoost (AB)
  - Logistic Regression (LR)
  - Support Vector Machine (SVM)
  - Neural Networks with 1, 2, or 3 hidden layers
- View prediction results with probabilities
- Automatic model training if needed

## Required Data Format

The CSV file should contain the following columns:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

Optionally, it can include an "Outcome" column (1 for diabetic, 0 for non-diabetic) for training purposes.

## Setup and Installation

### Prerequisites

- Node.js
- Python 3.8+
- npm or yarn

### Backend Setup

1. Install Python dependencies:
   ```
   cd backend
   pip install -r requirements.txt
   ```

2. Start the Flask server:
   ```
   python app.py
   ```

### Frontend Setup

1. Install Node.js dependencies:
   ```
   npm install
   ```

2. Start the development server:
   ```
   npm run dev
   ```

## Usage

1. Open the application in your browser
2. Select a machine learning model from the dropdown
3. Upload a CSV file with patient data
4. Click "Predict Diabetes" to get results
5. View the predictions and probabilities in the results section

## Sample Data

If you don't have your own data, you can use the Pima Indians Diabetes Dataset, which is automatically used for training if no model exists.