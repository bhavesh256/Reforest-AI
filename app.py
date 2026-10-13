
"""
Reforestation Intelligence Platform - Flask Backend
Serves the React SPA dashboard and REST API endpoints.
"""

import json
import logging
import math
import os
from pathlib import Path

from flask import (
    Flask,
    abort,
    jsonify,
    send_from_directory,
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "ml_outputs"
STATIC_DIR = BASE_DIR / "static"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask serves the frontend explicitly through the routes below.
app = Flask(__name__, static_folder=None)


# ============================================================
# JSON UTILITIES
# ============================================================

def sanitize_nan(obj):
    """Replace NaN and infinity with None for valid JSON."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj

    if isinstance(obj, dict):
        return {
            key: sanitize_nan(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [sanitize_nan(value) for value in obj]

    return obj


def load_json(filename):
    """Load JSON data from the ml_outputs directory."""
    filepath = OUTPUT_DIR / filename

    if not filepath.is_file():
        logger.error("Output file not found: %s", filepath)
        return {}

    try:
        with filepath.open("r", encoding="utf-8") as file:
            data = json.load(file)

        logger.info("Loaded output file: %s", filepath)
        return sanitize_nan(data)

    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to load output file: %s", filepath)
        return {}


# ============================================================
# RESPONSE HEADERS
# ============================================================

@app.after_request
def add_cache_headers(response):
    """Prevent stale dashboard and API responses from being cached."""
    response.headers["Cache-Control"] = (
        "no-store, no-cache, must-revalidate, max-age=0"
    )
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/healthz", methods=["GET"])
def healthz():
    """Health endpoint for Render."""
    return jsonify({"status": "healthy"}), 200


# ============================================================
# FRONTEND ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def index():
    """Serve the built React/Vite dashboard."""
    index_file = STATIC_DIR / "index.html"

    if not index_file.is_file():
        logger.error("Frontend index not found: %s", index_file)
        return "Frontend build missing. Check the Docker build.", 500

    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/<path:path>", methods=["GET"])
def frontend_files(path):
    """
    Serve frontend assets and React Router routes.
    API endpoints declared below take precedence over this route.
    """
    requested_file = STATIC_DIR / path

    if requested_file.is_file():
        return send_from_directory(str(STATIC_DIR), path)

    # Never return index.html for missing JavaScript, CSS, or images.
    if "." in path.split("/")[-1]:
        abort(404)

    # Fallback for client-side routes.
    index_file = STATIC_DIR / "index.html"

    if index_file.is_file():
        return send_from_directory(str(STATIC_DIR), "index.html")

    abort(404)


# ============================================================
# API ENDPOINTS
# ============================================================

@app.route("/api/overview", methods=["GET"])
def api_overview():
    """Country-level summary statistics."""
    return jsonify(load_json("overview.json"))


@app.route("/api/timeline", methods=["GET"])
def api_timeline():
    """Year-by-year national timeline."""
    return jsonify(load_json("national_timeline.json"))


@app.route("/api/states", methods=["GET"])
def api_states():
    """State-level data and rankings."""
    data = load_json("state_rankings.json")
    return jsonify(data)


@app.route("/api/state/<name>", methods=["GET"])
def api_state(name):
    """Detailed information and forecasts for a state."""
    states = load_json("state_rankings.json")
    forecasts = load_json("state_forecasts.json")

    if not isinstance(states, list):
        logger.error("State rankings must be a JSON list.")
        return jsonify({"error": "State data is unavailable"}), 503

    state_data = next(
        (
            state for state in states
            if isinstance(state, dict)
            and isinstance(state.get("name"), str)
            and state["name"].casefold() == name.casefold()
        ),
        None,
    )

    if state_data is None:
        return jsonify({"error": "State not found"}), 404

    if not isinstance(forecasts, dict):
        forecasts = {}

    state_data["forecast"] = forecasts.get(name, {})
    return jsonify(state_data)


@app.route("/api/predictions", methods=["GET"])
def api_predictions():
    """Future scenario predictions."""
    return jsonify({
        "state_forecasts": load_json("state_forecasts.json"),
        "country_forecast": load_json("country_forecast.json"),
    })


@app.route("/api/country-forecast", methods=["GET"])
def api_country_forecast():
    """Country-level forecast."""
    return jsonify(load_json("country_forecast.json"))


@app.route("/api/model-performance", methods=["GET"])
def api_model_performance():
    """Model metrics, feature importance, and SMOTE analysis."""
    return jsonify({
        "models": load_json("model_performance.json"),
        "feature_importance": load_json("feature_importance.json"),
        "smote_analysis": load_json("smote_analysis.json"),
    })


@app.route("/api/reforestation-plan", methods=["GET"])
def api_reforestation_plan():
    """Reforestation priorities and recommendations."""
    return jsonify(load_json("reforestation_plan.json"))


@app.route("/api/all", methods=["GET"])
def api_all():
    """Return all pipeline results."""
    return jsonify(load_json("pipeline_results.json"))


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":
    logger.info("Application directory: %s", BASE_DIR)
    logger.info("ML output directory: %s", OUTPUT_DIR)

    if not (OUTPUT_DIR / "pipeline_results.json").is_file():
        logger.warning(
            "Pipeline results are missing. Generate them before "
            "expecting populated dashboard data."
        )

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
