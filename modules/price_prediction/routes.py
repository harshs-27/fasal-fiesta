from flask import Blueprint, render_template, request, jsonify
from .model import predict_price, get_crops_and_states, SEASONS
from datetime import datetime

price_bp = Blueprint('price', __name__, template_folder='../../templates')

@price_bp.route('/')
def index():
    now = datetime.now()
    crops, states = get_crops_and_states()
    return render_template('price_prediction.html',
                           crops=crops, states=states, seasons=SEASONS,
                           current_month=now.month, current_year=now.year)

@price_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        result = predict_price(
            crop   = data['crop'],
            state  = data['state'],
            season = data['season'],
            month  = int(data['month']),
            year   = int(data['year']),
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400