"""
Plant Disease Detection — CNN with MobileNetV2 Transfer Learning
Dataset: PlantVillage (38 classes, ~87K images)
Download: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
           OR: https://data.mendeley.com/datasets/tywbtsjrjv/1
"""

import os
import numpy as np
from PIL import Image
import io

# ─────────────────────────────────────────────
#  Disease metadata: 38 PlantVillage classes
# ─────────────────────────────────────────────
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "display": "Apple — Apple Scab",
        "severity": "High",
        "treatment": "Apply fungicides (Captan / Mancozeb). Remove infected leaves. Prune for air circulation.",
        "prevention": "Use resistant varieties. Apply dormant sprays before bud break.",
        "color": "#c0392b"
    },
    "Apple___Black_rot": {
        "display": "Apple — Black Rot",
        "severity": "High",
        "treatment": "Remove mummified fruit. Apply Captan fungicide. Prune cankers.",
        "prevention": "Maintain sanitation. Avoid wounding trees.",
        "color": "#922b21"
    },
    "Apple___Cedar_apple_rust": {
        "display": "Apple — Cedar Apple Rust",
        "severity": "Medium",
        "treatment": "Apply Myclobutanil fungicide at first orange spots. Remove galls from cedars.",
        "prevention": "Plant resistant apple varieties. Keep cedars away from orchards.",
        "color": "#e67e22"
    },
    "Apple___healthy": {
        "display": "Apple — Healthy",
        "severity": "None",
        "treatment": "No treatment needed. Continue routine care.",
        "prevention": "Maintain proper irrigation, fertilization, and pruning schedule.",
        "color": "#27ae60"
    },
    "Blueberry___healthy": {
        "display": "Blueberry — Healthy",
        "severity": "None",
        "treatment": "No treatment needed.",
        "prevention": "Maintain soil pH 4.5–5.5. Proper mulching recommended.",
        "color": "#27ae60"
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "display": "Cherry — Powdery Mildew",
        "severity": "Medium",
        "treatment": "Apply Sulfur-based fungicide. Avoid overhead irrigation.",
        "prevention": "Improve air circulation. Plant in full sun.",
        "color": "#f39c12"
    },
    "Cherry_(including_sour)___healthy": {
        "display": "Cherry — Healthy",
        "severity": "None",
        "treatment": "No treatment needed.",
        "prevention": "Regular pruning and balanced fertilization.",
        "color": "#27ae60"
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "display": "Corn — Gray Leaf Spot",
        "severity": "High",
        "treatment": "Apply Azoxystrobin / Trifloxystrobin fungicide. Practice crop rotation.",
        "prevention": "Reduce crop residue. Plant resistant hybrids.",
        "color": "#c0392b"
    },
    "Corn_(maize)___Common_rust_": {
        "display": "Corn — Common Rust",
        "severity": "Medium",
        "treatment": "Apply Propiconazole fungicide. Early application is key.",
        "prevention": "Plant rust-resistant hybrids. Monitor fields weekly.",
        "color": "#e67e22"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "display": "Corn — Northern Leaf Blight",
        "severity": "High",
        "treatment": "Apply Mancozeb or Chlorothalonil. Destroy infected debris.",
        "prevention": "Rotate crops. Use certified disease-free seed.",
        "color": "#c0392b"
    },
    "Corn_(maize)___healthy": {
        "display": "Corn — Healthy",
        "severity": "None",
        "treatment": "No treatment needed.",
        "prevention": "Balanced NPK fertilization, adequate spacing.",
        "color": "#27ae60"
    },
    "Grape___Black_rot": {
        "display": "Grape — Black Rot",
        "severity": "High",
        "treatment": "Apply Myclobutanil or Mancozeb. Remove mummified berries.",
        "prevention": "Prune for air flow. Begin spray program at bud break.",
        "color": "#922b21"
    },
    "Grape___Esca_(Black_Measles)": {
        "display": "Grape — Esca (Black Measles)",
        "severity": "High",
        "treatment": "No effective chemical cure. Remove and destroy infected vines.",
        "prevention": "Protect pruning wounds. Use certified planting material.",
        "color": "#c0392b"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "display": "Grape — Leaf Blight",
        "severity": "Medium",
        "treatment": "Apply copper-based fungicides. Remove infected leaves.",
        "prevention": "Proper canopy management. Avoid water stress.",
        "color": "#e67e22"
    },
    "Grape___healthy": {
        "display": "Grape — Healthy",
        "severity": "None",
        "treatment": "No treatment needed.",
        "prevention": "Regular canopy management and irrigation scheduling.",
        "color": "#27ae60"
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "display": "Orange — Citrus Greening (HLB)",
        "severity": "Critical",
        "treatment": "No cure. Remove infected trees to prevent spread. Control psyllid insect vectors.",
        "prevention": "Use certified disease-free budwood. Regular insecticide for psyllids.",
        "color": "#7b241c"
    },
    "Peach___Bacterial_spot": {
        "display": "Peach — Bacterial Spot",
        "severity": "Medium",
        "treatment": "Apply copper-based bactericide. Avoid overhead irrigation.",
        "prevention": "Plant resistant varieties. Prune for air circulation.",
        "color": "#e67e22"
    },
    "Peach___healthy": {
        "display": "Peach — Healthy",
        "severity": "None",
        "treatment": "No treatment needed.",
        "prevention": "Annual dormant spray and proper thinning.",
        "color": "#27ae60"
    },
    "Pepper,_bell___Bacterial_spot": {
        "display": "Bell Pepper — Bacterial Spot",
        "severity": "Medium",
        "treatment": "Apply copper hydroxide spray. Remove infected plant parts.",
        "prevention": "Use disease-free seeds. Avoid working in wet fields.",
        "color": "#e67e22"
    },
    "Pepper,_bell___healthy": {
        "display": "Bell Pepper — Healthy",
        "severity": "None",
        "treatment": "No treatment needed.",
        "prevention": "Crop rotation and balanced irrigation.",
        "color": "#27ae60"
    },
    "Potato___Early_blight": {
        "display": "Potato — Early Blight",
        "severity": "Medium",
        "treatment": "Apply Chlorothalonil or Mancozeb. Remove lower infected leaves.",
        "prevention": "Maintain plant vigor. Rotate crops annually.",
        "color": "#e67e22"
    },
    "Potato___Late_blight": {
        "display": "Potato — Late Blight",
        "severity": "Critical",
        "treatment": "Apply Metalaxyl + Mancozeb immediately. Destroy infected tubers.",
        "prevention": "Use certified seed potatoes. Monitor humidity and temperature.",
        "color": "#7b241c"
    },
    "Potato___healthy": {
        "display": "Potato — Healthy",
        "severity": "None",
        "treatment": "No treatment needed.",
        "prevention": "Adequate hilling, irrigation management.",
        "color": "#27ae60"
    },
    "Raspberry___healthy": {
        "display": "Raspberry — Healthy",
        "severity": "None",
        "treatment": "No treatment needed.",
        "prevention": "Trellis support and annual cane removal.",
        "color": "#27ae60"
    },
    "Soybean___healthy": {
        "display": "Soybean — Healthy",
        "severity": "None",
        "treatment": "No treatment needed.",
        "prevention": "Inoculate seed with Rhizobium. Practice crop rotation.",
        "color": "#27ae60"
    },
    "Squash___Powdery_mildew": {
        "display": "Squash — Powdery Mildew",
        "severity": "Medium",
        "treatment": "Apply Neem oil or Potassium bicarbonate spray.",
        "prevention": "Plant in full sun. Space plants adequately.",
        "color": "#e67e22"
    },
    "Strawberry___Leaf_scorch": {
        "display": "Strawberry — Leaf Scorch",
        "severity": "Medium",
        "treatment": "Apply Captan fungicide. Remove infected leaves.",
        "prevention": "Use certified plants. Avoid overhead irrigation.",
        "color": "#e67e22"
    },
    "Strawberry___healthy": {
        "display": "Strawberry — Healthy",
        "severity": "None",
        "treatment": "No treatment needed.",
        "prevention": "Renovate beds annually after harvest.",
        "color": "#27ae60"
    },
    "Tomato___Bacterial_spot": {
        "display": "Tomato — Bacterial Spot",
        "severity": "High",
        "treatment": "Apply copper-based bactericide + Mancozeb. Remove infected plants.",
        "prevention": "Use disease-free seed. Stake plants for air circulation.",
        "color": "#c0392b"
    },
    "Tomato___Early_blight": {
        "display": "Tomato — Early Blight",
        "severity": "Medium",
        "treatment": "Apply Chlorothalonil spray. Remove infected lower leaves.",
        "prevention": "Mulch around plants. Rotate crops every 2–3 years.",
        "color": "#e67e22"
    },
    "Tomato___Late_blight": {
        "display": "Tomato — Late Blight",
        "severity": "Critical",
        "treatment": "Apply Metalaxyl immediately. Destroy infected plants urgently.",
        "prevention": "Monitor weather. Avoid overhead watering.",
        "color": "#7b241c"
    },
    "Tomato___Leaf_Mold": {
        "display": "Tomato — Leaf Mold",
        "severity": "Medium",
        "treatment": "Apply Chlorothalonil. Improve greenhouse ventilation.",
        "prevention": "Reduce humidity. Space plants well.",
        "color": "#e67e22"
    },
    "Tomato___Septoria_leaf_spot": {
        "display": "Tomato — Septoria Leaf Spot",
        "severity": "Medium",
        "treatment": "Apply Mancozeb or Copper fungicide. Remove infected leaves.",
        "prevention": "Mulch soil. Avoid working in wet fields.",
        "color": "#e67e22"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "display": "Tomato — Spider Mites",
        "severity": "Medium",
        "treatment": "Apply Miticide (Abamectin). Spray undersides of leaves with water.",
        "prevention": "Maintain humidity. Avoid water stress.",
        "color": "#e67e22"
    },
    "Tomato___Target_Spot": {
        "display": "Tomato — Target Spot",
        "severity": "High",
        "treatment": "Apply Chlorothalonil or Azoxystrobin fungicide.",
        "prevention": "Avoid excessive nitrogen. Ensure proper plant spacing.",
        "color": "#c0392b"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "display": "Tomato — Yellow Leaf Curl Virus",
        "severity": "Critical",
        "treatment": "No cure. Remove infected plants. Control whitefly vectors with Imidacloprid.",
        "prevention": "Use virus-resistant varieties. Yellow sticky traps for whiteflies.",
        "color": "#7b241c"
    },
    "Tomato___Tomato_mosaic_virus": {
        "display": "Tomato — Mosaic Virus",
        "severity": "High",
        "treatment": "No cure. Remove infected plants. Disinfect tools.",
        "prevention": "Use resistant varieties. Wash hands before handling plants.",
        "color": "#c0392b"
    },
    "Tomato___healthy": {
        "display": "Tomato — Healthy",
        "severity": "None",
        "treatment": "No treatment needed.",
        "prevention": "Regular monitoring and balanced fertilization.",
        "color": "#27ae60"
    },
}

CLASS_NAMES = list(DISEASE_INFO.keys())

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/disease_model.h5")


def build_and_train_model():
    """
    Call this script separately (python modules/disease_detection/model.py) to train.
    Requires PlantVillage dataset at: data/PlantVillage/
    """
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    IMG_SIZE   = 224
    BATCH_SIZE = 64
    EPOCHS     = 6
    DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/PlantVillage/color")

    print("Loading PlantVillage dataset from:", DATA_DIR)

    datagen = ImageDataGenerator(
        rescale=1./255, validation_split=0.2,
        rotation_range=20, width_shift_range=0.15,
        height_shift_range=0.15, horizontal_flip=True,
        zoom_range=0.15
    )
    train_gen = datagen.flow_from_directory(
        DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, subset='training', class_mode='categorical'
    )
    val_gen = datagen.flow_from_directory(
        DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, subset='validation', class_mode='categorical'
    )

    train_gen.samples = int(train_gen.samples * 0.4)
    val_gen.samples   = int(val_gen.samples * 0.4)

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False  # Freeze base first

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(len(CLASS_NAMES), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("Training frozen base...")
    model.fit(train_gen, validation_data=val_gen, epochs=5)

    # Fine-tune top layers
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])
    print("Fine-tuning top layers...")
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS - 5)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def load_disease_model():
    import tensorflow as tf
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Disease model not found at {MODEL_PATH}.\n"
            "Please train first: python modules/disease_detection/model.py\n"
            "After downloading PlantVillage dataset to data/PlantVillage/"
        )
    return tf.keras.models.load_model(MODEL_PATH)


def predict_disease(image_bytes):
    """Given raw image bytes, return top-3 disease predictions."""
    import tensorflow as tf

    model = load_disease_model()
    IMG_SIZE = 224

    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]
    top3_idx = np.argsort(preds)[::-1][:3]

    results = []
    for idx in top3_idx:
        cls_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "Unknown"
        info = DISEASE_INFO.get(cls_name, {
            "display": cls_name.replace('_', ' '),
            "severity": "Unknown",
            "treatment": "Consult local agronomist.",
            "prevention": "Monitor crops regularly.",
            "color": "#95a5a6"
        })
        results.append({
            "class": cls_name,
            "display": info["display"],
            "confidence": round(float(preds[idx]) * 100, 2),
            "severity": info["severity"],
            "treatment": info["treatment"],
            "prevention": info["prevention"],
            "color": info["color"],
        })

    return results


if __name__ == "__main__":
    build_and_train_model()
