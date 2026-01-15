from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv

load_dotenv()  # loads NANO_BANANA_API_KEY from .env

app = Flask(__name__)
CORS(app)  # Allow frontend JS to call backend

NANO_API_KEY = os.getenv("NANO_BANANA_API_KEY")

@app.route("/tryon", methods=["POST"])
def tryon():
    # 1️⃣ Get uploaded files
    person_file = request.files.get("person")
    garment_file = request.files.get("garment")

    if not person_file or not garment_file:
        return jsonify({"error": "Both person and garment images are required"}), 400

    # 2️⃣ Preset prompt
    prompt = (
        "Generate a realistic virtual try-on image by fitting the garment "
        "naturally on the person with correct proportions and lighting"
    )

    # 3️⃣ Prepare files for API
    files = {
        "person": (person_file.filename, person_file.stream, person_file.content_type),
        "garment": (garment_file.filename, garment_file.stream, garment_file.content_type)
    }

    # 4️⃣ Data and headers
    data = {"prompt": prompt}
    headers = {"Authorization": f"Bearer {NANO_API_KEY}"}

    # 5️⃣ Call Nano Banana API
    try:
        response = requests.post(
            "https://api.nanobanana.ai/v1/generate",
            headers=headers,
            files=files,
            data=data
        )
        response.raise_for_status()  # Raise error if status not 200

        result = response.json()

        # Example expected: {"result": "<URL-to-generated-image>"}
        return jsonify(result)

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Run backend locally on port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
