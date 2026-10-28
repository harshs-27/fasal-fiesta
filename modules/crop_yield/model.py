"""
Crop Yield Prediction + Fertilizer Recommendation
Dataset features: Temparature, Humidity, Moisture, Soil Type, Crop Type,
                  Nitrogen, Potassium, Phosphorous, Fertilizer, Yield, Rainfall
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

DATA_PATH    = os.path.join(os.path.dirname(__file__), "../../data/crop_yield_fertilizer.csv")
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "../../models")

YIELD_MODEL_PATH = os.path.join(MODEL_DIR, "yield_model.pkl")
FERT_MODEL_PATH  = os.path.join(MODEL_DIR, "fertilizer_model.pkl")
ENCODERS_PATH    = os.path.join(MODEL_DIR, "crop_yield_encoders.pkl")

SOIL_TYPES  = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
CROP_TYPES  = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy',
               'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']
FERTILIZERS = ['Urea', 'DAP', '14-35-14', '28-28', '17-17-17', '20-20', '10-26-26']


def train_models():
    """Train and save yield + fertilizer models."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    le_soil  = LabelEncoder()
    le_crop  = LabelEncoder()
    le_fert  = LabelEncoder()

    df['Soil Type enc']  = le_soil.fit_transform(df['Soil Type'])
    df['Crop Type enc']  = le_crop.fit_transform(df['Crop Type'])
    df['Fertilizer enc'] = le_fert.fit_transform(df['Fertilizer'])

    features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type enc', 'Crop Type enc',
                'Nitrogen', 'Potassium', 'Phosphorous', 'Rainfall']
    X = df[features]

    # --- Yield Regression ---
    y_yield = df['Yield']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_yield, test_size=0.2, random_state=42)
    yield_model = RandomForestRegressor(n_estimators=150, random_state=42)
    yield_model.fit(X_tr, y_tr)
    print(f"Yield model R² = {r2_score(y_te, yield_model.predict(X_te)):.4f}")

    # --- Fertilizer Classification ---
    y_fert = df['Fertilizer enc']
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y_fert, test_size=0.2, random_state=42)
    
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    fert_model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        ))
    ])
    fert_model.fit(X_tr2, y_tr2)
    print(f"Fertilizer model accuracy = {accuracy_score(y_te2, fert_model.predict(X_te2)):.4f}")

    # Save
    joblib.dump(yield_model, YIELD_MODEL_PATH)
    joblib.dump(fert_model,  FERT_MODEL_PATH)
    joblib.dump({'soil': le_soil, 'crop': le_crop, 'fert': le_fert}, ENCODERS_PATH)
    print("Models saved.")


def load_models():
    if not os.path.exists(YIELD_MODEL_PATH):
        print("Training models for the first time…")
        train_models()
    yield_model = joblib.load(YIELD_MODEL_PATH)
    fert_model  = joblib.load(FERT_MODEL_PATH)
    encoders    = joblib.load(ENCODERS_PATH)
    return yield_model, fert_model, encoders


def predict(temperature, humidity, moisture, soil_type, crop_type,
            nitrogen, potassium, phosphorous, rainfall):
    """Return yield prediction and fertilizer recommendation."""
    yield_model, fert_model, encoders = load_models()

    soil_enc = encoders['soil'].transform([soil_type])[0]
    crop_enc = encoders['crop'].transform([crop_type])[0]

    X = np.array([[temperature, humidity, moisture, soil_enc, crop_enc,
                   nitrogen, potassium, phosphorous, rainfall]])

    pred_yield = yield_model.predict(X)[0]
    pred_fert_enc = fert_model.predict(X)[0]
    pred_fert = encoders['fert'].inverse_transform([pred_fert_enc])[0]

    # Fertilizer probabilities for chart
    fert_probs = fert_model.predict_proba(X)[0]
    fert_classes = encoders['fert'].inverse_transform(fert_model.classes_)
    proba_dict = {cls: round(float(prob) * 100, 1) for cls, prob in zip(fert_classes, fert_probs)}

    return {
        "yield": round(float(pred_yield), 2),
        "fertilizer": pred_fert,
        "fertilizer_probabilities": proba_dict,
        "crop_type": crop_type,
        "soil_type": soil_type,
    }


def get_fertilizer_tip(fertilizer):
    tips = {
        "Urea":     "High nitrogen (46% N). Best for leafy crops. Apply in split doses to avoid nitrogen burn.",
        "DAP":      "Diammonium Phosphate (18-46-0). Excellent for root development and early growth stages.",
        "14-35-14": "Balanced NPK for flowering. Promotes strong root and fruit development.",
        "28-28":    "High N+P for vegetative growth phases. Ideal for nutrient-deficient soils.",
        "17-17-17": "Fully balanced NPK. Best for crops with equal macro-nutrient needs.",
        "20-20":    "Balanced starter fertilizer for seedbed preparation.",
        "10-26-26": "High P+K. Excellent for root crops and during pre-harvest phase.",
    }
    return tips.get(fertilizer, "Apply as per soil test recommendations.")


if __name__ == "__main__":
    train_models()
