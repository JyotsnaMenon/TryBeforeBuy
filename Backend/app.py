from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import base64
import requests
from PIL import Image
import io

# Load environment variables
load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
if not RAPIDAPI_KEY:
    raise ValueError("RAPIDAPI_KEY must be set in .env file!")

app = Flask(__name__)
CORS(app)

# RapidAPI settings
RAPIDAPI_HOST = "virtual-try-on7.p.rapidapi.com"
API_ENDPOINT = f"https://{RAPIDAPI_HOST}/results"


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Virtual Try-On API is running",
        "endpoints": {
            "/test": "GET",
            "/tryon": "POST"
        }
    })


@app.route("/test", methods=["GET"])
def test():
    return jsonify({
        "status": "ok",
        "api_key_set": bool(RAPIDAPI_KEY),
        "api_endpoint": API_ENDPOINT
    })


@app.route("/tryon", methods=["POST"])
def tryon():
    person_file = request.files.get("person")
    garment_file = request.files.get("garment")

    if not person_file or not garment_file:
        return jsonify({"error": "Upload both person and garment images"}), 400

    try:
        # Read images
        person_img = Image.open(io.BytesIO(person_file.read())).convert("RGB")
        garment_img = Image.open(io.BytesIO(garment_file.read())).convert("RGB")

        # Save to buffers
        person_buf = io.BytesIO()
        garment_buf = io.BytesIO()
        person_img.save(person_buf, format="PNG")
        garment_img.save(garment_buf, format="PNG")
        person_buf.seek(0)
        garment_buf.seek(0)

        print("Person image size:", len(person_buf.getvalue()))
        print("Garment image size:", len(garment_buf.getvalue()))

        headers = {
            "x-rapidapi-key": RAPIDAPI_KEY,
            "x-rapidapi-host": RAPIDAPI_HOST
        }

        files = {
            "image": ("person.png", person_buf.getvalue(), "image/png"),
            "image-apparel": ("garment.png", garment_buf.getvalue(), "image/png")
        }

        response = requests.post(
            API_ENDPOINT,
            headers=headers,
            files=files,
            timeout=120
        )

        print("Status:", response.status_code)
        print("Response:", response.text)

        # ---- SUCCESS ----
        if response.status_code == 200:
            content_type = response.headers.get("content-type", "").lower()

            # Case 1: direct image bytes
            if "image" in content_type:
                img_b64 = base64.b64encode(response.content).decode()
                return jsonify({"image": img_b64})

            # Case 2: JSON response (THIS IS YOUR CASE)
            if "application/json" in content_type:
                data = response.json()

                try:
                    result = data["results"][0]

                    if result["status"]["code"] != "ok":
                        return jsonify({"error": result}), 400

                    img_b64 = result["entities"][0]["image"]
                    return jsonify({"image": img_b64})

                except Exception as e:
                    return jsonify({
                        "error": "Failed to parse API response",
                        "details": str(e),
                        "raw": data
                    }), 500

            # Fallback
            img_b64 = base64.b64encode(response.content).decode()
            return jsonify({"image": img_b64})

        # ---- ERRORS ----
        if response.status_code == 429:
            return jsonify({"error": "Rate limit exceeded"}), 429

        if response.status_code == 403:
            return jsonify({"error": "Invalid API key"}), 403

        return jsonify({
            "error": f"API error {response.status_code}",
            "details": response.text
        }), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Server running at http://127.0.0.1:5000")
    app.run(debug=True)
