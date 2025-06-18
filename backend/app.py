from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)
CORS(app)

# Create models directory if it doesn't exist
os.makedirs('backend/models', exist_ok=True)

# Define feature names
FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

def preprocess_data(df):
    """Preprocess the data by handling missing values and scaling features."""
    # Replace 0 values with NaN for certain columns (except Pregnancies)
    for column in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        df[column] = df[column].replace(0, np.nan)
    
    # Fill missing values with median
    for column in df.columns:
        df[column] = df[column].fillna(df[column].median())
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    return X_scaled, scaler

def create_nn_model(hidden_layers=1):
    """Create a neural network model with specified number of hidden layers."""
    model = Sequential()
    
    # Input layer
    model.add(Dense(16, activation='relu', input_shape=(8,)))
    model.add(Dropout(0.2))
    
    # Hidden layers
    if hidden_layers >= 2:
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
    
    if hidden_layers >= 3:
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model(model_type, X, y):
    """Train a model based on the specified type."""
    if model_type == 'dt':
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_type == 'rf':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'nb':
        model = GaussianNB()
    elif model_type == 'ab':
        model = AdaBoostClassifier(random_state=42)
    elif model_type == 'lr':
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == 'svm':
        model = SVC(probability=True, random_state=42)
    elif model_type.startswith('nn'):
        # Extract number of hidden layers from model type (nn1, nn2, nn3)
        hidden_layers = int(model_type[2:])
        model = create_nn_model(hidden_layers)
        # For neural networks, we need to fit differently
        model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        return model
    
    # For scikit-learn models
    model.fit(X, y)
    return model

def get_predictions(model, X, model_type):
    """Get predictions and probabilities from the model."""
    if model_type.startswith('nn'):
        # For neural network models
        probabilities = model.predict(X).flatten()
        predictions = (probabilities > 0.5).astype(int)
    else:
        # For scikit-learn models
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]  # Probability of class 1
    
    return predictions, probabilities

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    model_type = request.form.get('model', 'dt')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Read the uploaded file
        content = file.read()
        
        # Check if the file contains training data with outcome or just features for prediction
        try:
            # Try to read as CSV with header
            df = pd.read_csv(io.BytesIO(content))
            
            # Check if the dataframe has the expected columns
            missing_columns = [col for col in FEATURE_NAMES if col not in df.columns]
            
            if missing_columns:
                return jsonify({
                    'error': f'Missing columns in CSV: {", ".join(missing_columns)}'
                }), 400
            
            # Check if 'Outcome' column exists (training data)
            is_training_data = 'Outcome' in df.columns
            
            # Extract features and target if available
            X = df[FEATURE_NAMES]
            y = df['Outcome'] if is_training_data else None
            
            # Preprocess the data
            X_scaled, scaler = preprocess_data(X)
            
            # Model file path
            model_path = f'backend/models/{model_type}_model.pkl'
            
            # If we have training data, train and save the model
            if is_training_data:
                # Split data for training and evaluation
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                
                # Train the model
                model = train_model(model_type, X_train, y_train)
                
                # Evaluate the model
                if model_type.startswith('nn'):
                    # For neural network models
                    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                    test_predictions, test_probabilities = get_predictions(model, X_scaled, model_type)
                else:
                    # For scikit-learn models
                    test_predictions = model.predict(X_test)
                    accuracy = accuracy_score(y_test, test_predictions)
                    test_predictions, test_probabilities = get_predictions(model, X_scaled, model_type)
                
                # Save the model and scaler
                if model_type.startswith('nn'):
                    # For neural network models, save in Keras format
                    model.save(model_path.replace('.pkl', '.h5'))
                else:
                    # For scikit-learn models
                    joblib.dump((model, scaler), model_path)
                
                # Return predictions and accuracy
                results = []
                for i, (pred, prob) in enumerate(zip(test_predictions, test_probabilities)):
                    results.append({
                        'prediction': int(pred),
                        'probability': float(prob),
                        'actual': int(y.iloc[i]) if i < len(y) else None
                    })
                
                return jsonify({
                    'accuracy': float(accuracy),
                    'predictions': results
                })
            
            else:  # Prediction only (no Outcome column)
                # Check if model exists
                if not os.path.exists(model_path) and not os.path.exists(model_path.replace('.pkl', '.h5')):
                    # If model doesn't exist, train on the Pima Indians Diabetes dataset
                    # Load the dataset
                    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
                    names = FEATURE_NAMES + ['Outcome']
                    dataset = pd.read_csv(url, names=names)
                    
                    # Preprocess the dataset
                    X_train = dataset[FEATURE_NAMES]
                    y_train = dataset['Outcome']
                    X_train_scaled, train_scaler = preprocess_data(X_train)
                    
                    # Train the model
                    model = train_model(model_type, X_train_scaled, y_train)
                    
                    # Save the model and scaler
                    if model_type.startswith('nn'):
                        # For neural network models, save in Keras format
                        model.save(model_path.replace('.pkl', '.h5'))
                        scaler = train_scaler  # Use the scaler from training
                    else:
                        # For scikit-learn models
                        joblib.dump((model, train_scaler), model_path)
                        scaler = train_scaler
                else:
                    # Load the existing model
                    if model_type.startswith('nn'):
                        from tensorflow.keras.models import load_model
                        model = load_model(model_path.replace('.pkl', '.h5'))
                        # We need to load the scaler separately or recreate it
                        # For simplicity, we'll just use the current scaler
                    else:
                        model, saved_scaler = joblib.load(model_path)
                        scaler = saved_scaler  # Use the saved scaler
                
                # Make predictions
                predictions, probabilities = get_predictions(model, X_scaled, model_type)
                
                # Return predictions
                results = []
                for pred, prob in zip(predictions, probabilities):
                    results.append({
                        'prediction': int(pred),
                        'probability': float(prob)
                    })
                
                return jsonify({
                    'predictions': results
                })
            
        except Exception as e:
            return jsonify({'error': f'Error processing CSV: {str(e)}'}), 400
    
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)