# training.py
from fastapi import UploadFile, File, Form
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

models = {
    "random_forest": RandomForestClassifier(),
    "logistic_regression": LogisticRegression()
}

trained_model = None
X_columns = []

def train_model(file: UploadFile = File(...), model_name: str = Form(...)):
    global trained_model, X_columns
    if model_name not in models:
        return {"error": "Invalid model name"}
    
    df = pd.read_csv(file.file)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_columns = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = models[model_name]
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    
    trained_model = model
    return {"message": "Model trained successfully", "metrics": metrics}
