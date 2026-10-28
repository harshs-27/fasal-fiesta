# 🌾 Fasal Fiesta — AI Companion for Cropping

> **Major Project** | Flask + Machine Learning | Python Backend + HTML/CSS/JS Frontend

---

## 📁 Project Structure

```
fasal_fiesta/
│
├── app.py                          ← 🚀 MAIN ENTRY POINT (run this)
├── requirements.txt                ← All Python dependencies
├── README.md
│
├── data/                           ← ALL DATASETS GO HERE
│   ├── crop_yield_fertilizer.csv   ← ✅ Already provided (your dataset)
│   ├── PlantVillage/               ← ⬇️ Download & place here (Module 2)
│   │   ├── Apple___Apple_scab/
│   │   ├── Apple___healthy/
│   │   └── ... (38 class folders)
│   └── crop_prices.csv             ← ⬇️ Download & place here (Module 3)
│                                      (auto-generated synthetic data if missing)
│
├── models/                         ← Saved ML models (auto-created on first run)
│   ├── yield_model.pkl
│   ├── fertilizer_model.pkl
│   ├── crop_yield_encoders.pkl
│   ├── price_model.pkl
│   ├── price_encoders.pkl
│   └── disease_model.h5
│
├── static/
│   ├── css/main.css               ← All styles (shared across modules)
│   └── js/main.js                 ← Animations & interactions
│
├── templates/
│   ├── index.html                 ← 🏠 Landing page
│   ├── crop_yield.html            ← 🌾 Module 1 UI
│   ├── disease_detection.html     ← 🔬 Module 2 UI
│   └── price_prediction.html      ← 📈 Module 3 UI
│
└── modules/
    ├── crop_yield/
    │   ├── model.py               ← RandomForest yield + fertilizer logic
    │   └── routes.py              ← Flask Blueprint routes
    │
    ├── disease_detection/
    │   ├── model.py               ← MobileNetV2 CNN disease classifier
    │   └── routes.py
    │
    └── price_prediction/
        ├── model.py               ← GradientBoosting price forecaster
        └── routes.py
```

---

## ⚡ Quick Start (Step-by-Step)

### Step 1 — Open in VS Code
```bash
code fasal_fiesta/
```

### Step 2 — Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install flask pandas numpy scikit-learn joblib pillow matplotlib plotly
# For Module 2 (Disease Detection):
pip install tensorflow
# For charts in frontend (already CDN, no pip needed)
```

### Step 4 — Run the App
```bash
python app.py
```
Open: **http://localhost:5000**

> Module 1 trains automatically on first use (takes ~30 seconds).
> Module 3 uses synthetic data automatically if dataset missing.
> Module 2 requires manual dataset download + training (see below).

---

## 📦 Datasets

### Module 1 — Crop Yield & Fertilizer
| | |
|---|---|
| **File** | `data/crop_yield_fertilizer.csv` |
| **Status** | ✅ Already provided |
| **Features** | Temperature, Humidity, Moisture, Soil Type, Crop Type, N, P, K, Rainfall, Fertilizer, Yield |
| **Records** | 8,000 |

---

### Module 2 — Plant Disease Detection
| | |
|---|---|
| **Dataset** | PlantVillage |
| **Images** | ~87,000 leaf images |
| **Classes** | 38 (14 crops, healthy + diseased) |
| **Download** | https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset |
| **Alt Link** | https://data.mendeley.com/datasets/tywbtsjrjv/1 |
| **Place at** | `data/PlantVillage/` (folder with 38 subfolders) |

**After downloading, train the model:**
```bash
python modules/disease_detection/model.py
# Training takes 1-3 hours on CPU, ~20 min on GPU
# Model saved to: models/disease_model.h5
```

> 💡 **Tip**: Use Google Colab with GPU for faster training. Then download `disease_model.h5` and place in `models/`.

---

### Module 3 — Price Prediction
| | |
|---|---|
| **Dataset Option 1** | Agmarknet (Government of India) |
| **Link** | https://data.gov.in/catalog/daily-commodity-wise-price-arrival-data-agmarknet |
| **Dataset Option 2** | Kaggle Agricultural Prices |
| **Link** | https://www.kaggle.com/datasets/kianwee/agricultural-raw-material-prices-1990-2020 |
| **Place at** | `data/crop_prices.csv` |
| **Auto-fallback** | ✅ Synthetic data auto-generated if file missing |

**Train with real data:**
```bash
python modules/price_prediction/model.py
```

---

## 🧠 Models Used

| Module | Algorithm | Library | Accuracy |
|--------|-----------|---------|---------|
| Yield Prediction | Random Forest Regressor | scikit-learn | ~R² 0.92+ |
| Fertilizer Recommendation | Random Forest Classifier | scikit-learn | ~95%+ |
| Disease Detection | MobileNetV2 (Transfer Learning) | TensorFlow/Keras | ~95% on PlantVillage |
| Price Prediction | Gradient Boosting Regressor | scikit-learn | Varies by crop |

---

## 🔧 VS Code Recommended Extensions
- **Python** (ms-python.python)
- **Pylance** (ms-python.vscode-pylance)
- **Thunder Client** (for API testing)
- **Live Server** (for HTML preview)
- **GitLens**

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: flask` | Run `pip install flask` in your venv |
| Models not found | Run each `model.py` file once to train |
| Disease model missing | Download PlantVillage, run `modules/disease_detection/model.py` |
| Port 5000 in use | Change port: `app.run(port=5001)` in `app.py` |
| TensorFlow install fails | Use `pip install tensorflow-cpu` for CPU-only |

---

## 🎨 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+, Flask |
| ML Models | scikit-learn, TensorFlow/Keras |
| Frontend | HTML5, CSS3 (custom design system), vanilla JS |
| Charts | Chart.js (CDN) |
| Fonts | Playfair Display, DM Sans, Space Mono (Google Fonts CDN) |
| Data | Pandas, NumPy |

---

## 📌 Module Route Map

| URL | Module |
|-----|--------|
| `/` | Home / Landing page |
| `/crop/` | Crop Yield & Fertilizer |
| `/crop/predict` | POST — returns yield + fertilizer JSON |
| `/disease/` | Plant Disease Detection |
| `/disease/predict` | POST — returns disease prediction JSON |
| `/price/` | Price Prediction |
| `/price/predict` | POST — returns price forecast JSON |

---

*Fasal Fiesta — Empowering Indian farmers with AI 🌾*
