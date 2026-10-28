"""
Crop Price Prediction — Rich Synthetic Dataset
Features: Crop, State, Month, Season, Year → Modal Price
Realistic price variation by state, month, season, year trend
"""

import os
import numpy as np
import pandas as pd
import joblib

DATA_PATH  = os.path.join(os.path.dirname(__file__), "../../data/crop_prices.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "../../models")
MODEL_PATH = os.path.join(MODEL_DIR, "price_model.pkl")
ENC_PATH   = os.path.join(MODEL_DIR, "price_encoders.pkl")
META_PATH  = os.path.join(MODEL_DIR, "price_meta.pkl")

CROPS = [
    "Wheat", "Rice", "Maize", "Tomato", "Potato", "Onion",
    "Cotton", "Sugarcane", "Soybean", "Groundnut", "Mustard",
    "Chilli", "Turmeric", "Garlic", "Banana", "Moong", "Arhar",
    "Barley", "Jowar", "Bajra"
]

STATES = [
    "Madhya Pradesh", "Uttar Pradesh", "Maharashtra", "Punjab", "Haryana",
    "Rajasthan", "Gujarat", "Karnataka", "Andhra Pradesh", "West Bengal",
    "Bihar", "Odisha", "Chhattisgarh", "Tamil Nadu", "Telangana"
]

SEASONS = ["Kharif", "Rabi", "Zaid", "Annual"]

# Base prices per crop (Rs/Quintal) — realistic MSP-based values
BASE_PRICES = {
    "Wheat": 2200, "Rice": 2000, "Maize": 1800, "Tomato": 1500,
    "Potato": 1000, "Onion": 1200, "Cotton": 6500, "Sugarcane": 400,
    "Soybean": 4400, "Groundnut": 5500, "Mustard": 5200,
    "Chilli": 8000, "Turmeric": 7000, "Garlic": 4000,
    "Banana": 1600, "Moong": 7500, "Arhar": 6600,
    "Barley": 1700, "Jowar": 2200, "Bajra": 2000
}

# State price multipliers (some states get higher/lower prices)
STATE_MULTIPLIERS = {
    "Punjab": 1.08, "Haryana": 1.06, "Madhya Pradesh": 0.95,
    "Uttar Pradesh": 0.97, "Maharashtra": 1.05, "Rajasthan": 0.93,
    "Gujarat": 1.03, "Karnataka": 1.04, "Andhra Pradesh": 1.02,
    "West Bengal": 0.98, "Bihar": 0.92, "Odisha": 0.91,
    "Chhattisgarh": 0.94, "Tamil Nadu": 1.06, "Telangana": 1.01
}

# Monthly price factors — captures seasonal demand/supply patterns
MONTHLY_FACTORS = {
    "Wheat":      [1.05,1.08,1.10,0.95,0.85,0.88,0.90,0.92,0.95,0.98,1.00,1.02],
    "Rice":       [1.00,1.02,1.05,1.08,1.10,1.05,0.90,0.88,0.85,0.90,0.95,0.98],
    "Maize":      [0.95,0.92,0.90,0.92,0.95,1.00,1.05,1.08,1.10,1.05,1.00,0.97],
    "Tomato":     [1.20,1.30,1.25,1.10,0.85,0.80,0.90,1.00,1.10,1.15,1.20,1.25],
    "Potato":     [1.10,1.05,0.95,0.85,0.80,0.85,0.90,0.95,1.00,1.05,1.10,1.12],
    "Onion":      [1.15,1.10,1.00,0.90,0.85,0.88,0.95,1.00,1.10,1.20,1.25,1.20],
    "Cotton":     [0.95,0.93,0.92,0.94,0.96,0.98,1.00,1.05,1.10,1.12,1.08,1.00],
    "Sugarcane":  [1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00],
    "Soybean":    [1.00,1.02,1.05,1.08,1.10,1.05,0.95,0.90,0.88,0.92,0.96,0.98],
    "Groundnut":  [0.95,0.93,0.92,0.94,0.98,1.02,1.05,1.08,1.10,1.08,1.05,1.00],
    "Mustard":    [1.08,1.10,1.05,0.95,0.90,0.88,0.90,0.93,0.96,1.00,1.03,1.06],
    "Chilli":     [1.10,1.15,1.20,1.15,1.05,0.95,0.90,0.88,0.92,0.98,1.05,1.08],
    "Turmeric":   [1.05,1.08,1.10,1.08,1.00,0.95,0.92,0.90,0.92,0.96,1.00,1.03],
    "Garlic":     [1.15,1.20,1.18,1.10,1.00,0.90,0.88,0.90,0.95,1.00,1.08,1.12],
    "Banana":     [1.00,1.02,1.05,1.08,1.10,1.08,1.00,0.95,0.93,0.95,0.98,1.00],
    "Moong":      [1.05,1.08,1.10,1.05,0.95,0.90,0.92,0.95,1.00,1.05,1.08,1.06],
    "Arhar":      [1.00,1.02,1.05,1.08,1.10,1.05,0.95,0.90,0.88,0.92,0.96,0.98],
    "Barley":     [1.05,1.08,1.10,0.95,0.88,0.90,0.92,0.95,0.98,1.00,1.02,1.04],
    "Jowar":      [0.95,0.93,0.92,0.95,1.00,1.05,1.08,1.10,1.08,1.05,1.00,0.97],
    "Bajra":      [0.95,0.93,0.92,0.95,1.00,1.05,1.08,1.10,1.08,1.05,1.00,0.97],
}

def get_season(month):
    if 6 <= month <= 10:  return "Kharif"
    elif 11 <= month or month <= 3: return "Rabi"
    elif 4 <= month <= 5: return "Zaid"
    else: return "Annual"


def generate_rich_dataset():
    """Generate 50,000 realistic records with proper variation."""
    np.random.seed(42)
    records = []

    for crop in CROPS:
        base = BASE_PRICES[crop]
        monthly = MONTHLY_FACTORS.get(crop, [1.0]*12)

        for state in STATES:
            state_mult = STATE_MULTIPLIERS.get(state, 1.0)

            for year in range(2018, 2026):
                year_trend = 1 + 0.04 * (year - 2018)  # 4% annual inflation

                for month in range(1, 13):
                    season = get_season(month)
                    m_factor = monthly[month - 1]

                    # Add realistic noise
                    noise = np.random.normal(1.0, 0.05)

                    price = base * state_mult * m_factor * year_trend * noise
                    price = max(base * 0.4, price)  # floor at 40% of base

                    records.append({
                        "Crop": crop,
                        "State": state,
                        "Month": month,
                        "Season": season,
                        "Year": year,
                        "Modal_Price": round(price, 2)
                    })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Generated {len(df)} records → saved to {DATA_PATH}")
    return df


def train_model():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Always regenerate with rich dataset
    df = generate_rich_dataset()

    le_crop   = LabelEncoder().fit(df['Crop'])
    le_state  = LabelEncoder().fit(df['State'])
    le_season = LabelEncoder().fit(df['Season'])

    df['crop_enc']   = le_crop.transform(df['Crop'])
    df['state_enc']  = le_state.transform(df['State'])
    df['season_enc'] = le_season.transform(df['Season'])

    FEATURES = ['crop_enc', 'state_enc', 'season_enc', 'Month', 'Year']
    X = df[FEATURES]
    y = df['Modal_Price']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.08,
        max_depth=6, random_state=42
    )
    model.fit(X_tr, y_tr)

    preds = model.predict(X_te)
    print(f"Price model — MAE: {mean_absolute_error(y_te, preds):.2f}, R²: {r2_score(y_te, preds):.4f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump({'crop': le_crop, 'state': le_state, 'season': le_season}, ENC_PATH)
    joblib.dump({'features': FEATURES}, META_PATH)
    print("Price model saved.")


def load_price_model():
    if not os.path.exists(MODEL_PATH):
        print("Training price model for the first time...")
        train_model()
    model    = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENC_PATH)
    meta     = joblib.load(META_PATH)
    return model, encoders, meta


def predict_price(crop, state, season, month, year):
    model, encoders, meta = load_price_model()

    def safe_transform(enc, val):
        val = str(val).strip().title()
        if val in enc.classes_:
            return enc.transform([val])[0]
        return 0

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    trend   = []
    labels  = []

    for i in range(12):
        m = (int(month) + i - 1) % 12 + 1
        y = int(year) + (int(month) + i - 1) // 12
        s = get_season(m)

        X = np.array([[
            safe_transform(encoders['crop'], crop),
            safe_transform(encoders['state'], state),
            safe_transform(encoders['season'], s),
            m, y
        ]])
        price = max(0, float(model.predict(X)[0]))
        trend.append(round(price, 2))
        labels.append(month_names[m-1] + f" '{str(y)[2:]}")

    base_price = trend[0]

    return {
        "crop":            crop.title(),
        "state":           state.title(),
        "season":          season,
        "predicted_price": base_price,
        "trend_labels":    labels,
        "trend_prices":    trend,
        "min_price":       round(min(trend), 2),
        "max_price":       round(max(trend), 2),
        "best_sell_month": labels[trend.index(max(trend))],
        "unit":            "₹ / Quintal"
    }


def get_crops_and_states():
    return sorted(CROPS), sorted(STATES)


if __name__ == "__main__":
    train_model()