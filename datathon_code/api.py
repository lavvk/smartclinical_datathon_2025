"""
FastAPI backend for SmartClinical
Provides endpoints for risk prediction and resource allocation
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import pandas as pd
import joblib
import os
from datetime import datetime

# Load model on startup
MODEL_PATH = "risk_model.joblib"
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup"""
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Warning: Model file {MODEL_PATH} not found. Please train the model first.")
    yield
    # Cleanup (if needed)

app = FastAPI(
    title="SmartClinical API",
    description="Risk-Driven Hospital Resource Allocation System",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class PatientVitals(BaseModel):
    Respiratory_Rate: float = Field(..., ge=0, le=60, description="Respiratory rate per minute")
    Oxygen_Saturation: float = Field(..., ge=0, le=100, description="Oxygen saturation percentage")
    O2_Scale: int = Field(..., ge=1, le=3, description="O2 scale (1-3)")
    Systolic_BP: float = Field(..., ge=0, le=300, description="Systolic blood pressure")
    Heart_Rate: float = Field(..., ge=0, le=300, description="Heart rate per minute")
    Temperature: float = Field(..., ge=30, le=45, description="Body temperature in Celsius")
    Consciousness: str = Field(..., description="Consciousness level: A (Alert), P (Pain), U (Unresponsive), V (Verbal)")
    On_Oxygen: int = Field(..., ge=0, le=1, description="On oxygen: 0 (No), 1 (Yes)")

class PredictionRequest(BaseModel):
    patient: PatientVitals

class PredictionResponse(BaseModel):
    risk_level: str
    risk_score: float
    probabilities: Dict[str, float]

class AllocationPatient(BaseModel):
    patient_id: Optional[str] = None
    Respiratory_Rate: float
    Oxygen_Saturation: float
    O2_Scale: int
    Systolic_BP: float
    Heart_Rate: float
    Temperature: float
    Consciousness: str
    On_Oxygen: int

class AllocationRequest(BaseModel):
    patients: List[AllocationPatient]
    available_oxygen: int = Field(..., ge=0, description="Number of available oxygen units")
    available_staff: int = Field(..., ge=0, description="Number of available staff members")

class AllocatedPatient(BaseModel):
    patient_id: str
    risk_level: str
    risk_score: float
    probabilities: Dict[str, float]
    allocated_oxygen: bool
    allocated_staff: bool
    vitals: Dict[str, float]

class AllocationResponse(BaseModel):
    allocation: List[AllocatedPatient]
    summary: Dict[str, int]
    timestamp: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SmartClinical API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "Single patient risk prediction",
            "/allocate": "Batch resource allocation"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(request: PredictionRequest):
    """
    Predict risk level for a single patient
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        # Convert to DataFrame
        patient_dict = request.patient.dict()
        df = pd.DataFrame([patient_dict])
        
        # Predict
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        
        # Get class names and create probability dict
        class_names = model.classes_
        prob_dict = {label: float(prob) for label, prob in zip(class_names, probabilities)}
        
        # Get risk score (probability of High risk, or max if High not available)
        risk_score = prob_dict.get("High", max(prob_dict.values()))
        
        return PredictionResponse(
            risk_level=prediction,
            risk_score=float(risk_score),
            probabilities=prob_dict
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/allocate", response_model=AllocationResponse)
async def allocate_resources(request: AllocationRequest):
    """
    Allocate resources to a batch of patients based on risk predictions
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        # Predict risk for all patients
        patients_data = []
        for i, patient in enumerate(request.patients):
            patient_dict = {
                "Respiratory_Rate": patient.Respiratory_Rate,
                "Oxygen_Saturation": patient.Oxygen_Saturation,
                "O2_Scale": patient.O2_Scale,
                "Systolic_BP": patient.Systolic_BP,
                "Heart_Rate": patient.Heart_Rate,
                "Temperature": patient.Temperature,
                "Consciousness": patient.Consciousness,
                "On_Oxygen": patient.On_Oxygen,
            }
            patients_data.append({
                "patient_id": patient.patient_id or f"P{i+1:04d}",
                "vitals": patient_dict,
                "patient_obj": patient
            })
        
        # Batch prediction
        df = pd.DataFrame([p["vitals"] for p in patients_data])
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)
        class_names = model.classes_
        
        # Mapping for Consciousness: A (Alert) -> 0, P (Painful) -> 1, V (Verbal) -> 2, U (Unresponsive) -> 3
        consciousness_map = {"A": 0.0, "P": 1.0, "V": 2.0, "U": 3.0}
        
        # Create patient records with predictions
        patient_records = []
        for i, p_data in enumerate(patients_data):
            prob_dict = {label: float(prob) for label, prob in zip(class_names, probabilities[i])}
            risk_level = predictions[i]
            risk_score = prob_dict.get("High", max(prob_dict.values()))
            
            # Convert vitals for response (Consciousness needs to be numeric)
            vitals_for_response = p_data["vitals"].copy()
            consciousness_str = vitals_for_response["Consciousness"]
            vitals_for_response["Consciousness"] = consciousness_map.get(consciousness_str, 0.0)
            
            patient_records.append({
                "patient_id": p_data["patient_id"],
                "risk_level": risk_level,
                "risk_score": float(risk_score),
                "probabilities": prob_dict,
                "vitals": vitals_for_response
            })
        
        # Sort by risk (High > Medium > Low > Normal)
        risk_priority = {"High": 4, "Medium": 3, "Low": 2, "Normal": 1}
        patient_records.sort(
            key=lambda x: (risk_priority.get(x["risk_level"], 0), x["risk_score"]),
            reverse=True
        )
        
        # Allocate resources
        # Priority: High and Medium risk patients get oxygen first
        # High risk patients get staff first
        oxygen_allocated = 0
        staff_allocated = 0
        
        for patient in patient_records:
            patient["allocated_oxygen"] = False
            patient["allocated_staff"] = False
            
            # Allocate oxygen to High/Medium risk patients
            if patient["risk_level"] in ["High", "Medium"]:
                if oxygen_allocated < request.available_oxygen:
                    patient["allocated_oxygen"] = True
                    oxygen_allocated += 1
            
            # Allocate staff to High risk patients
            if patient["risk_level"] == "High":
                if staff_allocated < request.available_staff:
                    patient["allocated_staff"] = True
                    staff_allocated += 1
        
        # Create response
        allocated_patients = [
            AllocatedPatient(
                patient_id=p["patient_id"],
                risk_level=p["risk_level"],
                risk_score=p["risk_score"],
                probabilities=p["probabilities"],
                allocated_oxygen=p["allocated_oxygen"],
                allocated_staff=p["allocated_staff"],
                vitals=p["vitals"]
            )
            for p in patient_records
        ]
        
        # Summary statistics
        summary = {
            "total_patients": len(patient_records),
            "high_risk": sum(1 for p in patient_records if p["risk_level"] == "High"),
            "medium_risk": sum(1 for p in patient_records if p["risk_level"] == "Medium"),
            "low_risk": sum(1 for p in patient_records if p["risk_level"] == "Low"),
            "normal": sum(1 for p in patient_records if p["risk_level"] == "Normal"),
            "oxygen_allocated": oxygen_allocated,
            "staff_allocated": staff_allocated,
            "available_oxygen": request.available_oxygen,
            "available_staff": request.available_staff
        }
        
        return AllocationResponse(
            allocation=allocated_patients,
            summary=summary,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Allocation error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

