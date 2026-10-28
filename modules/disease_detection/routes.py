from flask import Blueprint, render_template, request, jsonify
from .model import predict_disease, DISEASE_INFO, CLASS_NAMES

disease_bp = Blueprint('disease', __name__, template_folder='../../templates')


@disease_bp.route('/')
def index():
    return render_template('disease_detection.html')


@disease_bp.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'}), 400

    try:
        image_bytes = file.read()
        results = predict_disease(image_bytes)
        return jsonify({'success': True, 'predictions': results})
    except FileNotFoundError as e:
        return jsonify({'success': False, 'error': str(e)}), 503
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
