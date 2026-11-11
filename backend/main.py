
from pathlib import Path
import pickle
import sqlite3
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model.pkl"
DB_PATH = BASE_DIR / "predictions.db"

FEATURES = [
    "avg_usage",
    "peak_usage",
    "off_peak_usage",
    "variance"
]

app = FastAPI(
    title="Energy Theft Detection API",
    description="Machine Learning powered electricity theft detection",
    version="1.0.0"
)

# Load your existing trained model once
if not MODEL_PATH.exists():
    raise RuntimeError("model.pkl not found. Train your model first.")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# Input validation
class EnergyData(BaseModel):
    avg_usage: float = Field(ge=0, allow_inf_nan=False)
    peak_usage: float = Field(ge=0, allow_inf_nan=False)
    off_peak_usage: float = Field(ge=0, allow_inf_nan=False)
    variance: float = Field(ge=0, allow_inf_nan=False)


class BatchEnergyData(BaseModel):
    records: list[EnergyData] = Field(min_length=1, max_length=10000)


# Database initialization
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                avg REAL,
                peak REAL,
                off_peak REAL,
                variance REAL,
                prediction INTEGER,
                theft_score REAL
            )
        """)


init_db()


# Convert input into model features
def prepare_features(data):
    return pd.DataFrame(
        [[
            data.avg_usage,
            data.peak_usage,
            data.off_peak_usage,
            data.variance
        ]],
        columns=FEATURES
    )


# Get theft probability
def get_theft_score(features):
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Model does not support probability prediction.")

    classes = list(model.classes_)

    if 1 not in classes:
        raise RuntimeError("Theft class 1 is missing from the model.")

    probabilities = model.predict_proba(features)[0]

    return float(probabilities[classes.index(1)])


# Save prediction
def save_prediction(data, prediction, theft_score):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO prediction_history (
                timestamp, avg, peak, off_peak,
                variance, prediction, theft_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(timespec="seconds"),
            data.avg_usage,
            data.peak_usage,
            data.off_peak_usage,
            data.variance,
            prediction,
            theft_score * 100
        ))


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Energy Theft Detection API is running",
        "model": "Random Forest",
        "status": "Active"
    }


# Health endpoint
@app.get("/health")
def health():
    return {"status": "healthy"}


# Individual prediction
@app.post("/predict")
def predict(data: EnergyData):

    try:
        features = prepare_features(data)

        prediction = int(model.predict(features)[0])
        theft_score = get_theft_score(features)

        save_prediction(data, prediction, theft_score)

        return {
            "prediction": prediction,
            "prediction_label": (
                "Theft Detected"
                if prediction == 1
                else "Normal Usage"
            ),
            "theft_score": round(theft_score * 100, 2)
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# Batch prediction
@app.post("/predict/batch")
def predict_batch(data: BatchEnergyData):

    try:
        features = pd.DataFrame(
            [record.model_dump() for record in data.records],
            columns=FEATURES
        )

        predictions = model.predict(features)
        probabilities = model.predict_proba(features)

        classes = list(model.classes_)

        if 1 not in classes:
            raise ValueError("Theft class 1 is missing from the model.")

        theft_index = classes.index(1)

        results = []

        timestamp = datetime.now().isoformat(timespec="seconds")
        rows = []

        for i in range(len(data.records)):

            prediction = int(predictions[i])
            score = float(probabilities[i][theft_index]) * 100

            record = data.records[i]

            results.append({
                **record.model_dump(),
                "prediction": prediction,
                "prediction_label": (
                    "Theft Detected"
                    if prediction == 1
                    else "Normal Usage"
                ),
                "theft_score": round(score, 2)
            })

            rows.append((
                timestamp,
                record.avg_usage,
                record.peak_usage,
                record.off_peak_usage,
                record.variance,
                prediction,
                score
            ))

        with sqlite3.connect(DB_PATH) as conn:
            conn.executemany("""
                INSERT INTO prediction_history (
                    timestamp, avg, peak, off_peak,
                    variance, prediction, theft_score
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rows)

        return {
            "total_records": len(results),
            "theft_cases": sum(
                1 for item in results
                if item["prediction"] == 1
            ),
            "results": results
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# Prediction history
@app.get("/history")
def history(limit: int = 100):

    if not 1 <= limit <= 1000:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 1000."
        )

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        records = conn.execute("""
            SELECT *
            FROM prediction_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return {
        "total_records": len(records),
        "history": [dict(record) for record in records]
    }
