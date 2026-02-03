from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import time
from PIL import Image
from pillow_heif import register_heif_opener
import numpy as np

from backend.modules import get_card_data_from_image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
register_heif_opener()

@app.route("/api/fetch_price", methods=["POST"])
def analyze():
    """
    Analyze an uploaded image and return price statistics.
    
    Expects: multipart/form-data with 'image' field
    Returns: JSON with price stats
    """

    print("Starting")

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_raw = request.files["image"]
    
    if image_raw.filename == "":
        return jsonify({"error": "No image selected"}), 400
    
    image = np.array(Image.open(image_raw))    
    
    data = get_card_data_from_image(image=image)

    if data["object"] != "card":
        print("s")
        return jsonify({"status": "card_read_error"})

    rarity = data["rarity"]
    price = data["prices"]["usd"]
    price_foil = data["prices"]["usd_foil"]
    collector_num = data["collector_number"]
    
    stats = {
        "status": "success",
        "rarity": rarity,
        "price": f"${price}",
        "price_foil": f"${price_foil}",
        "collector_num": collector_num.capitalize(),
    }
    
    return jsonify(stats)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("Starting Flask server on http://localhost:5000")
    print("Endpoints:")
    print("  POST /analyze - Upload image for price analysis")
    print("  GET /health - Health check")
    app.run(debug=True, port=5000)
