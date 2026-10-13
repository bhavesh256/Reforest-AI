
"""
Reforestation Intelligence Platform - Flask Backend
Serves the React/Vite dashboard and REST API endpoints.
"""

import json
import logging
import math
import os
from pathlib import Path

from flask import Flask, abort, jsonify, send_from_directory

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "ml_outputs"
STATIC_DIR = BASE_DIR / "static"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Serve frontend files through explicit routes.
app = Flask(__name__, static_folder=None)


# ============================================================
# JSON UTILITIES
# ============================================================

def sanitize_nan(obj):
    """Replace NaN and infinity with None for valid JSON."""
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj

    if isinstance(obj, dict):
        return {key: sanitize_nan(value) for key, value in obj.items()}

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
            return sanitize_nan(json.load(file))
    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to load JSON file: %s", filepath)
        return {}


# ============================================================
# RESPONSE HEADERS
# ============================================================

@app.after_request
def add_cache_headers(response):
    """Prevent stale API and frontend responses from being cached."""
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
    return jsonify({"status": "healthy"}), 200


# ============================================================
# FRONTEND ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def index():
    """Serve the Vite-generated HTML."""
    if not (STATIC_DIR / "index.html").is_file():
        logger.error("Frontend index missing: %s", STATIC_DIR / "index.html")
        return "Frontend build missing. Check the Docker build.", 500

    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/static/<path:filename>", methods=["GET"])
def static_files(filename):
    """Serve Vite JavaScript, CSS, and other static assets."""
    return send_from_directory(str(STATIC_DIR), filename)


# ============================================================
# API ENDPOINTS
# ============================================================

@app.route("/api/overview", methods=["GET"])
def api_overview():
    return jsonify(load_json("overview.json"))


@app.route("/api/timeline", methods=["GET"])
def api_timeline():
    return jsonify(load_json("national_timeline.json"))


@app.route("/api/states", methods=["GET"])
def api_states():
    return jsonify(load_json("state_rankings.json"))


@app.route("/api/state/<name>", methods=["GET"])
def api_state(name):
    states = load_json("state_rankings.json")
    forecasts = load_json("state_forecasts.json")

    if not isinstance(states, list):
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

    state_data["forecast"] = (
        forecasts.get(name, {}) if isinstance(forecasts, dict) else {}
    )
    return jsonify(state_data)


@app.route("/api/predictions", methods=["GET"])
def api_predictions():
    return jsonify({
        "state_forecasts": load_json("state_forecasts.json"),
        "country_forecast": load_json("country_forecast.json"),
    })


@app.route("/api/country-forecast", methods=["GET"])
def api_country_forecast():
    return jsonify(load_json("country_forecast.json"))


@app.route("/api/model-performance", methods=["GET"])
def api_model_performance():
    return jsonify({
        "models": load_json("model_performance.json"),
        "feature_importance": load_json("feature_importance.json"),
        "smote_analysis": load_json("smote_analysis.json"),
    })


@app.route("/api/reforestation-plan", methods=["GET"])
def api_reforestation_plan():
    return jsonify(load_json("reforestation_plan.json"))


@app.route("/api/all", methods=["GET"])
def api_all():
    return jsonify(load_json("pipeline_results.json"))


# ============================================================
# FRONTEND ROUTE FALLBACK
# Keep this AFTER all API routes.
# ============================================================

@app.route("/<path:path>", methods=["GET"])
def frontend_fallback(path):
    """
    Support React client-side routes without hiding missing assets.
    API routes above take precedence.
    """
    if path.startswith(("api/", "static/")):
        abort(404)

    # Do not return HTML for missing JS, CSS, image, or other files.
    if "." in path.split("/")[-1]:
        abort(404)

    if (STATIC_DIR / "index.html").is_file():
        return send_from_directory(str(STATIC_DIR), "index.html")

    abort(404)


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":
    logger.info("Application directory: %s", BASE_DIR)
    logger.info("ML output directory: %s", OUTPUT_DIR)

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
