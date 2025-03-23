from pydantic import BaseModel

class PredictionInput(BaseModel):
    features: list

def predict_model(input_data: PredictionInput):
    global trained_model, X_columns
    if trained_model is None:
        return {"error": "No trained model found"}
    
    if len(input_data.features) != len(X_columns):
        return {"error": "Incorrect number of features"}
    
    prediction = trained_model.predict([input_data.features])
    return {"prediction": prediction.tolist()}