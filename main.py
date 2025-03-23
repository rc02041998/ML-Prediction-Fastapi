from fastapi import FastAPI
from training import train_model
from prediction import predict_model

app = FastAPI()

app.post("/train/")(train_model)
app.post("/predict/")(predict_model)