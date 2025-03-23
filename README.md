# ML Prediction System using FastAPI

This project provides a simple machine learning prediction system using FastAPI. Users can upload a dataset, train a model, and make predictions through API endpoints.

## Features
- Upload a CSV dataset for training.
- Choose between `RandomForestClassifier` and `LogisticRegression`.
- Train the selected model and store it for later predictions.
- Make predictions by sending input data through an API request.
- Return classification results based on trained models.

## Installation

### Prerequisites
Ensure you have Python installed (>= 3.8) and create a virtual environment:

```bash
python -m venv myenv
source myenv/bin/activate  # On Windows use: myenv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the FastAPI Server

```bash
uvicorn main:app --reload
```

The API will be available at: `http://127.0.0.1:8000`

## API Endpoints

### 1. Train Model
**Endpoint:** `POST /train/`

**Description:** Upload a CSV file and specify the model to train.

**Request:**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/train/' \
  -F 'file=@dataset.csv' \
  -F 'model_name=random_forest'
```

**Response:**
```json
{
  "message": "Model trained successfully",
  "metrics": {
    "accuracy": 0.95,
    "precision": 0.96,
    "recall": 0.94,
    "f1_score": 0.95
  }
}
```

### 2. Predict Data
**Endpoint:** `POST /predict/`

**Description:** Send input data to get predictions from the trained model.

**Request:**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'Content-Type: application/json' \
  -d '{
    "Id": 13.0,
    "SepalLengthCm": 4.8,
    "SepalWidthCm": 3.0,
    "PetalLengthCm": 1.4,
    "PetalWidthCm": 0.1
  }'
```

**Response:**
```json
{
  "prediction": ["Iris-setosa"]
}
```

## Notes
- Ensure the dataset includes the correct column structure used during training.
- Trained models are temporary and lost after restarting the server.

