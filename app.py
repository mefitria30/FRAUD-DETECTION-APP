"""
Fraud Detection API with FastAPI
=================================
Fixed version dengan better error handling
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from datetime import datetime

# ============================================================================
# Create FastAPI App FIRST (before loading models)
# ============================================================================

app = FastAPI(
    title="Fraud Detection API",
    description="API untuk deteksi fraud/kecurangan menggunakan KNN model dan TF-IDF",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global variables for lazy loading
# ============================================================================

model = None
tfidf_vectorizer = None
model_loaded = False

# ============================================================================
# Load Models Function (Lazy Loading)
# ============================================================================

def load_models():
    """
    Load models lazily when first request comes
    """
    global model, tfidf_vectorizer, model_loaded
    
    if model_loaded:
        return True
    
    try:
        import joblib
        
        # Check if model files exist
        if not os.path.exists('knn_model.pkl'):
            raise FileNotFoundError("knn_model.pkl not found!")
        
        if not os.path.exists('tfidf_vectorizer.pkl'):
            raise FileNotFoundError("tfidf_vectorizer.pkl not found!")
        
        # Load models
        print("Loading models...")
        model = joblib.load('knn_model.pkl')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        model_loaded = True
        print("✅ Models loaded successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to load models: {str(e)}"
        )

# ============================================================================
# Request/Response Models
# ============================================================================

class TextInput(BaseModel):
    text: str = Field(..., description="Teks percakapan yang akan diprediksi")

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., description="List teks untuk prediksi batch")

class PredictionResponse(BaseModel):
    text: str
    prediction: str
    is_fraud: bool
    fraud_probability: float
    label: int
    timestamp: str

class BatchPredictionResponse(BaseModel):
    count: int
    results: List[PredictionResponse]
    timestamp: str

# ============================================================================
# Helper Functions
# ============================================================================

def predict_single_text(text: str) -> dict:
    """
    Predict fraud for a single text
    """
    # Ensure models are loaded
    load_models()
    
    # Transform text
    text_tfidf = tfidf_vectorizer.transform([text])
    
    # Predict
    prediction = model.predict(text_tfidf)[0]
    proba = model.predict_proba(text_tfidf)[0]
    
    # Calculate fraud probability
    is_fraud = bool(prediction == 1)
    fraud_prob = float(proba[1]) if len(proba) > 1 else 0.0
    
    return {
        "text": text,
        "prediction": "FRAUD" if is_fraud else "NORMAL",
        "is_fraud": is_fraud,
        "fraud_probability": round(fraud_prob * 100, 2),
        "label": int(prediction),
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
def root():
    """Root endpoint - informasi API"""
    return {
        "name": "Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model_loaded,
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint"""
    try:
        load_models()
        return {
            "status": "healthy",
            "model_loaded": model_loaded,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "model_loaded": False
        }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_fraud(input: TextInput):
    """
    Prediksi fraud untuk single text
    """
    try:
        result = predict_single_text(input.text)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(input: BatchTextInput):
    """
    Prediksi fraud untuk multiple texts
    """
    try:
        results = []
        
        for text in input.texts:
            result = predict_single_text(text)
            results.append(result)
        
        return {
            "count": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/model/info", tags=["Model Info"])
def model_info():
    """Informasi tentang model yang digunakan"""
    try:
        load_models()
        
        return {
            "model_type": type(model).__name__,
            "n_neighbors": model.n_neighbors if hasattr(model, 'n_neighbors') else None,
            "vectorizer_type": type(tfidf_vectorizer).__name__,
            "vocabulary_size": len(tfidf_vectorizer.vocabulary_) if hasattr(tfidf_vectorizer, 'vocabulary_') else None,
            "model_loaded": model_loaded
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    try:
        load_models()
    except Exception as e:
        print(f"⚠️ Warning: Could not load models on startup: {e}")
        print("Models will be loaded on first request")

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 Starting Fraud Detection API Server")
    print("="*70)
    print("\n📡 Server will be available at:")
    print("   - http://localhost:5000")
    print("\n📚 API Documentation:")
    print("   - http://localhost:5000/docs")
    print("\n" + "="*70 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )