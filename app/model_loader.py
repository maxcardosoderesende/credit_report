import joblib
import os

# store the model and feature columns as module-level globals. 
# They start as None and get populated at startup:

_model = None
_feature_columns = None

def load_model():
 """Load model and feature list ONCE at startup."""
 global _model, _feature_columns
 
 model_path = os.environ.get("MODEL_PATH", "model/credit_model.pkl")
 features_path = os.environ.get("FEATURES_PATH", "model/feature_columns.pkl")
 
 if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")
 
 _model = joblib.load(model_path)
 _feature_columns = joblib.load(features_path)
 print(f"Model loaded from {model_path} ({len(_feature_columns)} features)")
 return _model


def get_model():
    if _model is None:
        raise RuntimeError("Model not loaded! Call load_model() first.")
    return _model

def get_feature_columns():
    if _feature_columns is None:
        raise RuntimeError("Features not loaded!")
    return _feature_columns