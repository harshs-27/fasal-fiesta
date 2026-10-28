from flask import Flask, render_template, request, jsonify, redirect, url_for
from modules.crop_yield.routes import crop_yield_bp
from modules.disease_detection.routes import disease_bp
from modules.price_prediction.routes import price_bp

app = Flask(__name__)
app.secret_key = "fasal_fiesta_2024"

# Register blueprints
app.register_blueprint(crop_yield_bp, url_prefix="/crop")
app.register_blueprint(disease_bp, url_prefix="/disease")
app.register_blueprint(price_bp, url_prefix="/price")

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
