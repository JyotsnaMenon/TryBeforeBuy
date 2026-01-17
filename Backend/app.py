from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os, base64, requests
from PIL import Image
import io

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
if not RAPIDAPI_KEY:
    raise ValueError("RAPIDAPI_KEY must be set in .env file!")

app = Flask(__name__)
CORS(app)

# Try-On Diffusion API settings
RAPIDAPI_HOST = "try-on-diffusion.p.rapidapi.com"
API_ENDPOINT = f"https://{RAPIDAPI_HOST}/try-on-file"

@app.route("/", methods=["GET"])
def home():
    """Home route"""
    return jsonify({
        "message": "Virtual Try-On API is running!",
        "endpoints": {
            "/test": "GET - Test API status",
            "/tryon": "POST - Generate virtual try-on"
        }
    })

@app.route("/test", methods=["GET"])
def test():
    """Test endpoint to verify API key and server status"""
    return jsonify({
        "status": "Server running ✅",
        "api_key_set": bool(RAPIDAPI_KEY),
        "api_key_preview": RAPIDAPI_KEY[:10] + "..." if RAPIDAPI_KEY else "Not set",
        "api_endpoint": API_ENDPOINT,
        "api_host": RAPIDAPI_HOST
    })

@app.route("/tryon", methods=["POST"])
def tryon():
    person_file = request.files.get("person")
    garment_file = request.files.get("garment")

    if not person_file or not garment_file:
        return jsonify({"error": "Upload both images"}), 400

    try:
        # Read images
        person_bytes = person_file.read()
        garment_bytes = garment_file.read()

        # Open and validate images
        person_img = Image.open(io.BytesIO(person_bytes))
        garment_img = Image.open(io.BytesIO(garment_bytes))

        # Convert to RGB
        if person_img.mode != 'RGB':
            person_img = person_img.convert('RGB')
        if garment_img.mode != 'RGB':
            garment_img = garment_img.convert('RGB')

        # Convert to PNG format with good quality to ensure file size is adequate
        person_buffer = io.BytesIO()
        garment_buffer = io.BytesIO()
        person_img.save(person_buffer, format='PNG')
        garment_img.save(garment_buffer, format='PNG')
        person_buffer.seek(0)
        garment_buffer.seek(0)

        print(f"Person image size: {len(person_buffer.getvalue())} bytes")
        print(f"Garment image size: {len(garment_buffer.getvalue())} bytes")

        # Prepare headers for RapidAPI
        # Note: Don't set Content-Type for multipart/form-data - requests will set it automatically
        headers = {
            "x-rapidapi-key": RAPIDAPI_KEY,
            "x-rapidapi-host": RAPIDAPI_HOST
        }

        # Prepare form data with correct field names from API documentation
        files = {
            'clothing_image': ('clothing.png', garment_buffer.getvalue(), 'image/png'),
            'avatar_image': ('avatar.png', person_buffer.getvalue(), 'image/png')
        }

        # Optional parameters according to API docs
        data = {}  # Start with empty, API says all params are optional

        print(f"Calling RapidAPI: {API_ENDPOINT}")
        print(f"Using API Key: {RAPIDAPI_KEY[:10]}...")
        print(f"Files being sent: {list(files.keys())}")
        print(f"Data parameters: {data}")
        
        # Make request to Try-On Diffusion API
        response = requests.post(
            API_ENDPOINT,
            headers=headers,
            files=files,
            data=data,
            timeout=120  # Try-on can take time
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Body: {response.text}")  # Print full response body

        if response.status_code == 200:
            content_type = response.headers.get('content-type', '').lower()
            
            # Check if response is an image
            if 'image' in content_type:
                # Response is direct image bytes
                img_b64 = base64.b64encode(response.content).decode('utf-8')
                print("Success! Image received.")
                return jsonify({"image": img_b64})
            
            # Check if response is JSON
            elif 'application/json' in content_type:
                data = response.json()
                print(f"JSON Response: {data}")
                
                # Try different possible response formats
                if "image" in data:
                    return jsonify({"image": data["image"]})
                elif "result" in data:
                    return jsonify({"image": data["result"]})
                elif "output" in data:
                    return jsonify({"image": data["output"]})
                elif "url" in data:
                    # Download image from URL
                    img_response = requests.get(data["url"])
                    img_b64 = base64.b64encode(img_response.content).decode('utf-8')
                    return jsonify({"image": img_b64})
                else:
                    return jsonify({
                        "error": f"Unexpected JSON format. Keys: {list(data.keys())}",
                        "debug": data
                    }), 500
            else:
                # Unknown content type, try to treat as image
                try:
                    img_b64 = base64.b64encode(response.content).decode('utf-8')
                    return jsonify({"image": img_b64})
                except:
                    return jsonify({
                        "error": f"Unexpected content type: {content_type}",
                        "response_preview": response.text[:500]
                    }), 500

        elif response.status_code == 429:
            return jsonify({
                "error": "Rate limit exceeded. You've used up your free API calls. Please wait or upgrade your plan."
            }), 429
        
        elif response.status_code == 403:
            return jsonify({
                "error": "Authentication failed. Please check your API key in the .env file."
            }), 403
        
        elif response.status_code == 400:
            try:
                error_data = response.json()
                error_msg = error_data.get('message', error_data.get('error', 'Invalid parameters'))
                print(f"400 Error Details: {error_data}")
                return jsonify({
                    "error": f"Bad request: {error_msg}",
                    "details": error_data
                }), 400
            except:
                print(f"400 Error Raw: {response.text}")
                return jsonify({
                    "error": f"Bad request: {response.text[:300]}"
                }), 400
        
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get('message', str(error_data))
            except:
                error_msg = response.text[:500]
            
            print(f"Error Response: {error_msg}")
            return jsonify({
                "error": f"API Error ({response.status_code}): {error_msg}"
            }), response.status_code

    except requests.exceptions.Timeout:
        return jsonify({
            "error": "Request timed out. The API took too long to respond. Please try again."
        }), 504
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

if __name__ == "__main__":
    print("Starting Virtual Try-On API Server...")
    print("Server will be available at: http://127.0.0.1:5000")
    print("Test endpoint: http://127.0.0.1:5000/test")
    app.run(debug=True)