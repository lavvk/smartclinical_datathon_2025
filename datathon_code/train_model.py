"""
Model training script for SmartClinical
Trains the RandomForest model and saves it for use in the API
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

def load_data(path="Health_Risk_Dataset.csv"):
    """Load and prepare the dataset"""
    df = pd.read_csv(path)
    X = df.drop(columns=["Risk_Level", "Patient_ID"])
    y = df["Risk_Level"]
    return X, y

def build_pipeline():
    """Build the preprocessing and model pipeline"""
    numeric_features = ["Respiratory_Rate", "Oxygen_Saturation", "O2_Scale", 
                       "Systolic_BP", "Heart_Rate", "Temperature", "On_Oxygen"]
    categories = ["Consciousness"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categories),
            ("num", "passthrough", numeric_features),
        ]
    )
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )
    
    clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    return clf

def train_model():
    """Train the model and save it"""
    print("Loading data...")
    X, y = load_data()
    
    print("Building pipeline...")
    clf = build_pipeline()
    
    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print("Training model...")
    clf.fit(X_train, y_train)
    
    print("Evaluating model...")
    preds = clf.predict(X_val)
    print("\nClassification Report:")
    print(classification_report(y_val, preds))
    
    # Save the model
    model_path = "risk_model.joblib"
    joblib.dump(clf, model_path)
    print(f"\nModel saved to {model_path}")
    
    return clf

if __name__ == "__main__":
    if not os.path.exists("Health_Risk_Dataset.csv"):
        print("Error: Health_Risk_Dataset.csv not found!")
        exit(1)
    
    train_model()
    print("\nTraining complete!")

