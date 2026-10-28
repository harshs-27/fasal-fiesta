from flask import Blueprint, render_template, request, jsonify
from .model import predict, get_fertilizer_tip, SOIL_TYPES, CROP_TYPES

crop_yield_bp = Blueprint('crop_yield', __name__, template_folder='../../templates')


@crop_yield_bp.route('/')
def index():
    return render_template('crop_yield.html', soil_types=SOIL_TYPES, crop_types=CROP_TYPES)


@crop_yield_bp.route('/predict', methods=['POST'])
def predict_yield():
    try:
        data = request.json
        result = predict(
            temperature  = float(data['temperature']),
            humidity     = float(data['humidity']),
            moisture     = float(data['moisture']),
            soil_type    = data['soil_type'],
            crop_type    = data['crop_type'],
            nitrogen     = float(data['nitrogen']),
            potassium    = float(data['potassium']),
            phosphorous  = float(data['phosphorous']),
            rainfall     = float(data['rainfall']),
        )
        result['fertilizer_tip'] = get_fertilizer_tip(result['fertilizer'])
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400
