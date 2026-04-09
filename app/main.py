# =========================
# BUILDING THE FASTAPI APPLICATION
# =========================
# Wiring together the model loader, the validation schemas, and the HTTP endpoints.
# This is the actual application that will run in production, receiving requests and returning predictions.

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import time
import logging
import json
from datetime import datetime, timezone

# Load the two functions I created in previous steps
from app.schemas import LoanApplication, PredictionResponse, HealthResponse
from app.model_loader import load_model, get_model, get_feature_columns


# =========================
# LOGGING CONFIGURATION
# =========================
# Every prediction the API makes should be recorded.
# These logs are the foundation of monitoring — they let us detect drift, track performance, and debug issues.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("credit_scoring_api")


# =========================
# LIFECYCLE MANAGEMENT
# =========================
# We need the model to be loaded before the first request arrives
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, clean up at shutdown."""
    
    logger.info("Starting up — loading model...")
    load_model()  # Load model into memory
    
    logger.info("Ready to serve predictions.")
    
    yield  # The app runs while we're "inside" the yield
    
    logger.info("Shutting down.")


# =========================
# FASTAPI APP INSTANCE
# =========================
app = FastAPI(
    title="Home Credit Scoring API",
    description="Predict loan default probability using the Home Credit dataset",
    version="1.0.0",
    lifespan=lifespan,
)


# =========================
# HEALTH CHECK ENDPOINT
# =========================
# Verify that a deployment succeeded
# CI/CD pipeline uses it to verify that a deployment succeeded.
# It’s a simple “are you there?” ping.

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        model = get_model()
        return HealthResponse(status="healthy", model_loaded=True)
    
    except RuntimeError:
        return HealthResponse(status="unhealthy", model_loaded=False)


# =========================
# PREDICTION ENDPOINT
# =========================
# What clients call to get predictions
# It receives a loan application, validates it (Pydantic does this automatically),
# computes the feature ratios, runs the model, logs everything, and returns the result.
@app.post("/predict", response_model=PredictionResponse)
async def predict(application: LoanApplication):
    
    start_time = time.time()
    
    try:
        model = get_model()
        feature_columns = get_feature_columns()

        # =========================
        # BUILD FEATURE DICTIONARY
        # =========================
        # IMPORTANT: Keep naming EXACTLY as training data
        features = {
            'EXT_SOURCE_1': application.ext_source_1,
            'EXT_SOURCE_2': application.ext_source_2,
            'EXT_SOURCE_3': application.ext_source_3,
            'AMT_INCOME_TOTAL': application.amt_income_total,
            'AMT_CREDIT': application.amt_credit,
            'AMT_ANNUITY': application.amt_annuity,
            'AMT_GOODS_PRICE': application.amt_goods_price,
            'CODE_GENDER': application.code_gender,
            'FLAG_OWN_CAR': application.flag_own_car,
            'FLAG_OWN_REALTY': application.flag_own_realty,
            'CNT_CHILDREN': application.cnt_children,
            'AGE_YEARS': application.age_years,
            'YEARS_EMPLOYED': application.years_employed,
            'YEARS_ID_PUBLISH': application.years_id_publish,
            'EDUCATION_LEVEL': application.education_level,

            # Engineered features (ONLY HERE — not again later)
            'CREDIT_INCOME_RATIO': application.amt_credit / (application.amt_income_total + 1),
            'ANNUITY_INCOME_RATIO': application.amt_annuity / (application.amt_income_total + 1),
            'CREDIT_GOODS_RATIO': application.amt_credit / (application.amt_goods_price + 1),
        }

        # =========================
        # CREATE DATAFRAME SAFELY
        # =========================
        input_data = pd.DataFrame([features])

        # Ensure correct order + missing columns handled
        input_data = input_data.reindex(columns=feature_columns, fill_value=0)

        # =========================
        # MODEL PREDICTION
        # =========================
        probability = model.predict_proba(input_data)[0][1]
        prediction = int(probability > 0.5)

        # =========================
        # RISK LOGIC
        # =========================
        if probability < 0.3:
            risk = "Low"
        elif probability < 0.6:
            risk = "Medium"
        else:
            risk = "High"

        # =========================
        # LOGGING
        # =========================
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": application.model_dump(),
            "prediction": prediction,
            "probability": probability,
            "risk": risk,
            "latency_ms": round((time.time() - start_time) * 1000, 2),
        }

        logger.info(json.dumps(log_entry))

        # =========================
        # RESPONSE
        # =========================
        return PredictionResponse(
            prediction=prediction,
            probability_of_default=round(float(probability), 4),
            risk_category=risk,
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail="Internal Server Error during prediction",
        )