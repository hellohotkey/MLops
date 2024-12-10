import os
import pickle

from pydantic import BaseModel, conlist
from typing import List
from fastapi import FastAPI, Body

with open("model.pkl", "rb") as f:
        model = pickle.load(f)

class Dataset(BaseModel):
    data: List

app = FastAPI()


@app.post("/predict")
def get_prediction(dataset: Dataset):
    data = dict(dataset)["data"]
    prediction = model.predict(data).tolist()
    log_proba = model.predict_proba(data).tolist()
    return {"prediction": prediction, "log_proba": log_proba}


if __name__ == "__main__":
    print("test")
