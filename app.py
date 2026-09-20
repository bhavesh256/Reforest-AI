"""
Reforestation Web Application - Flask Backend
==============================================
Serves the SPA dashboard and REST API endpoints for ML pipeline results.
"""

import json
import os
from pathlib import Path
from flask import Flask, render_template, jsonify, send_from_directory

app = Flask(__name__)

OUTPUT_DIR = Path('ml_outputs')


def load_json(filename):
    """Load a JSON file from the ml_outputs directory."""
    filepath = OUTPUT_DIR / filename
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the main SPA dashboard."""
    return render_template('index.html')


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/overview')
def api_overview():
    """Country-level summary statistics."""
    return jsonify(load_json('overview.json'))


@app.route('/api/timeline')
def api_timeline():
    """Year-by-year national timeline data."""
    return jsonify(load_json('national_timeline.json'))


@app.route('/api/states')
def api_states():
    """State-level data with rankings."""
    return jsonify(load_json('state_rankings.json'))


@app.route('/api/state/<name>')
def api_state(name):
    """Detailed data for a specific state."""
    states = load_json('state_rankings.json')
    forecasts = load_json('state_forecasts.json')

    state_data = next((s for s in states if s['name'] == name), None)
    if state_data is None:
        return jsonify({'error': 'State not found'}), 404

    state_data['forecast'] = forecasts.get(name, {})
    return jsonify(state_data)


@app.route('/api/predictions')
def api_predictions():
    """Future scenario predictions."""
    return jsonify({
        'state_forecasts': load_json('state_forecasts.json'),
        'country_forecast': load_json('country_forecast.json'),
    })


@app.route('/api/country-forecast')
def api_country_forecast():
    """Country-level forecast."""
    return jsonify(load_json('country_forecast.json'))


@app.route('/api/model-performance')
def api_model_performance():
    """ML model metrics and comparisons."""
    return jsonify({
        'models': load_json('model_performance.json'),
        'feature_importance': load_json('feature_importance.json'),
        'smote_analysis': load_json('smote_analysis.json'),
    })


@app.route('/api/reforestation-plan')
def api_reforestation_plan():
    """Reforestation priority areas and recommendations."""
    return jsonify(load_json('reforestation_plan.json'))


@app.route('/api/all')
def api_all():
    """All pipeline results in one call."""
    return jsonify(load_json('pipeline_results.json'))


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    # Check if ML pipeline has been run
    if not (OUTPUT_DIR / 'pipeline_results.json').exists():
        print("⚠️  ML pipeline results not found. Running pipeline first...")
        from ml_pipeline import ReforestationPipeline
        pipeline = ReforestationPipeline()
        pipeline.run()
        print("✅ Pipeline complete. Starting web server...")
    else:
        print("✅ ML pipeline results found. Starting web server...")

    app.run(debug=True, host='0.0.0.0', port=5000)
