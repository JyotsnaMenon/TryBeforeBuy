import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import base64
import requests
from PIL import Image
import io
import pickle
import numpy as np
import cv2
from sklearn.cluster import KMeans
import colorsys
import json    
import random
from deepface import DeepFace


load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

if not RAPIDAPI_KEY:
    raise ValueError("RAPIDAPI_KEY missing in .env")

RAPIDAPI_HOST = "virtual-try-on7.p.rapidapi.com"
API_ENDPOINT = f"https://{RAPIDAPI_HOST}/results"

# --- NEW: Merchant API Key for B2B Access ---
MERCHANT_API_KEY = os.getenv("VTO_MERCHANT_KEY", "trybeforebuy_secret_123")

def require_apikey(req):
    """Simple check to ensure the merchant is authorized"""
    return req.headers.get("X-API-KEY") == MERCHANT_API_KEY

try:
    with open("models/color_model.pkl", "rb") as f:
        color_model = pickle.load(f)
    with open("models/style_model.pkl", "rb") as f:
        style_model = pickle.load(f)
    with open("models/overall_model.pkl", "rb") as f:
        overall_model = pickle.load(f)
    with open("models/label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open("models/feature_info.pkl", "rb") as f:
        feature_info = pickle.load(f)
    print("✅ All models loaded successfully")
except Exception as e:
    print(f"❌ Model loading error: {e}")
    raise


# ---Load Recommendation Dataset ---
try:
    with open("dataset.json", "r") as f:
        MERCHANT_CATALOG = json.load(f)
    print(f"✅ Loaded {len(MERCHANT_CATALOG)} items from dataset.json")
except Exception as e:
    print(f"⚠️ Could not load dataset: {e}")
    MERCHANT_CATALOG = []

app = Flask(__name__)
CORS(app)

LAST_RESULT_IMAGE = None
LAST_PERSON_IMAGE = None

# --------------------------
# Feature Extraction Functions (Unchanged)
# --------------------------
def extract_dominant_color_kmeans(image_b64):
    try:
        img_data = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img = np.array(pil_img) 
        
        img = cv2.resize(img, (300, 300))
        height, width = img.shape[:2]
        
        crop_y1 = int(height * 0.30)
        crop_y2 = int(height * 0.80)
        crop_x1 = int(width * 0.25)
        crop_x2 = int(width * 0.75)
        
        garment_region = img[crop_y1:crop_y2, crop_x1:crop_x2]
        lab_img = cv2.cvtColor(garment_region, cv2.COLOR_RGB2LAB)
        pixels = lab_img.reshape(-1, 3)
        
        l_channel = pixels[:, 0]
        valid_mask = (l_channel > 20) & (l_channel < 235)
        filtered_pixels = pixels[valid_mask]
        
        if len(filtered_pixels) < 50:
            filtered_pixels = pixels
            
        n_colors = 3
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(filtered_pixels)
        
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        counts = np.bincount(labels)
        
        dominant_idx = np.argmax(counts)
        dominant_lab = centers[dominant_idx]
        
        dominant_lab_pixel = np.uint8([[dominant_lab]])
        dominant_rgb_pixel = cv2.cvtColor(dominant_lab_pixel, cv2.COLOR_LAB2RGB)
        dominant_rgb = dominant_rgb_pixel[0][0]
        
        print(f"  📊 Dominant RGB: {dominant_rgb}")
        color_name = rgb_to_color_name_advanced(dominant_rgb)
        return color_name
        
    except Exception as e:
        print(f"❌ Color extraction error: {e}")
        return "Blue"

def rgb_to_color_name_advanced(rgb):
    r, g, b = rgb
    r_norm, g_norm, b_norm = r/255.0, g/255.0, b/255.0
    h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    h_deg = h * 360
    
    if s < 0.18:
        if v > 0.85: return "White"
        elif v > 0.60: return "Grey"
        elif v > 0.25: return "Grey"
        else: return "Black"
    
    if s < 0.35 and v > 0.75:
        if v > 0.88: return "White"
        if h_deg < 30 or h_deg >= 330: return "Pink"
        elif h_deg < 60: return "Yellow"
        elif h_deg < 150: return "Green"
        elif h_deg < 270: return "Blue"
        else: return "Pink"
    
    if h_deg < 10 or h_deg >= 350: return "Red"
    elif 10 <= h_deg < 25: return "Red" if s > 0.5 else "Orange"
    elif 25 <= h_deg < 50: return "Orange"
    elif 50 <= h_deg < 70: return "Brown" if v < 0.50 else "Yellow"
    elif 70 <= h_deg < 165: return "Green"
    elif 165 <= h_deg < 200: return "Cyan"
    elif 200 <= h_deg < 250: return "Navy" if v < 0.45 or (s > 0.6 and v < 0.60) else "Blue"
    elif 250 <= h_deg < 295: return "Purple"
    elif 295 <= h_deg < 330: return "Pink"
    elif 330 <= h_deg < 350: return "Pink" if v > 0.6 else "Red"
    else: return "Grey"

def detect_skin_tone_advanced(image_b64):
    try:
        img_data = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img_rgb = np.array(pil_img)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        scale_factor = 400 / max(img.shape[:2])
        new_width = int(img.shape[1] * scale_factor)
        new_height = int(img.shape[0] * scale_factor)
        img_resized = cv2.resize(img, (new_width, new_height))
        img_rgb_resized = cv2.resize(img_rgb, (new_width, new_height))
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        skin_pixels = []
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                forehead_y1 = y + int(h * 0.2)
                forehead_y2 = y + int(h * 0.4)
                forehead_x1 = x + int(w * 0.3)
                forehead_x2 = x + int(w * 0.7)
                face_region = img_rgb_resized[forehead_y1:forehead_y2, forehead_x1:forehead_x2]
                if face_region.size > 0:
                    skin_pixels = extract_skin_pixels_ycrcb(face_region)
        
        if len(skin_pixels) == 0:
            height, width = img_rgb_resized.shape[:2]
            regions = [
                img_rgb_resized[int(height*0.15):int(height*0.35), int(width*0.35):int(width*0.65)],
                img_rgb_resized[int(height*0.30):int(height*0.50), int(width*0.30):int(width*0.70)],
            ]
            for region in regions:
                if region.size > 0:
                    pixels = extract_skin_pixels_ycrcb(region)
                    skin_pixels.extend(pixels)
        
        if len(skin_pixels) < 10:
            return "medium"
        
        skin_pixels = np.array(skin_pixels)
        avg_skin_rgb = np.mean(skin_pixels, axis=0)
        skin_tone = calculate_skin_tone_ita(avg_skin_rgb)
        return skin_tone
        
    except Exception as e:
        print(f"❌ Skin tone detection error: {e}")
        return "medium"

def extract_skin_pixels_ycrcb(img_region):
    ycrcb = cv2.cvtColor(img_region, cv2.COLOR_RGB2YCR_CB)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_pixels = img_region[skin_mask > 0]
    return skin_pixels.tolist()

def calculate_skin_tone_ita(rgb):
    r, g, b = rgb
    rgb_pixel = np.uint8([[[r, g, b]]])
    lab_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2LAB)
    l, a, b_lab = lab_pixel[0][0]
    
    if b_lab == 0: b_lab = 0.001
    ita = np.arctan((l - 50) / b_lab) * (180 / np.pi)
    
    if ita > 55: return "light"
    elif ita > 41: return "light"
    elif ita > 28: return "medium"
    elif ita > 19: return "medium"
    elif ita > 10: return "tan"
    else: return "dark"

def infer_style_from_occasion(occasion):
    style_map = {
        "casual": "casual",
        "business": "formal",
        "formal": "formal",
        "party": "modern",
        "sports": "sporty",
        "gym": "sporty"
    }
    return style_map.get(occasion.lower(), "casual")


def detect_demographic(image_b64, fallback_age_group="women", fallback_gender="female"):
    """Uses DeepFace AI with a Garment-Data Fallback for failures"""
    try:
        if not image_b64:
            raise ValueError("No image provided")
            
        print("🤖 AI is analyzing face for Demographics...")
        img_data = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(img_data)).convert('RGB')
        
        # 1. Resize the image BEFORE giving it to DeepFace! Massive memory saver.
        pil_img.thumbnail((400, 400)) 
        
        img_rgb = np.array(pil_img)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # 2. Force a garbage collection BEFORE the heavy math starts
        import gc
        gc.collect() 

        result = DeepFace.analyze(
            img, 
            actions=['age', 'gender'], 
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        # 3. Force garbage collection AFTER the math is done
        gc.collect()

        res = result[0] if isinstance(result, list) else result
        
        age = res['age']
        dominant_gender = res['dominant_gender'] 
        
        print(f"👤 AI Raw Output: {age} yrs, {dominant_gender}")
        gender = "male" if dominant_gender == "Man" else "female"

        if age <= 10: 
            age_group = "boys" if gender == "male" else "girls"
        else:
            age_group = "men" if gender == "male" else "women"
            
        return age_group, gender
            
    except Exception as e:
        print(f"⚠️ Face detection failed ({e}). Using Garment Fallback: {fallback_age_group}, {fallback_gender}")
        return fallback_age_group, fallback_gender

# --------------------------
# Encode features safely & Comments
# --------------------------
def encode_input_safe(occasion, color, skin, style):
    try:
        available_occasions = list(label_encoders['occasion'].classes_)
        available_colors = list(label_encoders['color_simple'].classes_)
        available_skins = list(label_encoders['skin_tone'].classes_)
        available_styles = list(label_encoders['style_category'].classes_)
        
        if occasion not in available_occasions: occasion = 'casual' if 'casual' in available_occasions else available_occasions[0]
        if color not in available_colors:
            color_lower = color.lower()
            for avail_color in available_colors:
                if avail_color.lower() == color_lower:
                    color = avail_color
                    break
            else:
                color = 'Blue' if 'Blue' in available_colors else available_colors[0]
                
        if skin not in available_skins: skin = 'medium' if 'medium' in available_skins else available_skins[0]
        if style not in available_styles: style = 'casual' if 'casual' in available_styles else available_styles[0]
        
        occ_enc = label_encoders["occasion"].transform([occasion])[0]
        col_enc = label_encoders["color_simple"].transform([color])[0]
        ski_enc = label_encoders["skin_tone"].transform([skin])[0]
        sty_enc = label_encoders["style_category"].transform([style])[0]
        
        return np.array([[occ_enc, col_enc, ski_enc, sty_enc]])
        
    except Exception as e:
        print(f"❌ Encoding error: {e}")
        return None

def generate_comment(color_rating, style_rating, overall_rating, occasion):
    if overall_rating >= 9: main = random.choice(["Absolutely stunning!", "Perfect match!"])
    elif overall_rating >= 8: main = random.choice(["Great choice!", "Looks fantastic!"])
    elif overall_rating >= 7: main = random.choice(["Good choice!", "Nice look!"])
    elif overall_rating >= 6: main = random.choice(["Acceptable.", "It works."])
    elif overall_rating >= 5: main = random.choice(["Average match.", "Could be improved."])
    elif overall_rating >= 4: main = random.choice(["Below average.", "Not the best choice."])
    else: main = random.choice(["Poor match.", "Try different options."])
    
    feedback_parts = []
    if color_rating >= 8: feedback_parts.append("The color suits you beautifully.")
    elif color_rating < 5: feedback_parts.append("Consider trying different colors.")
    if style_rating >= 8: feedback_parts.append(f"Perfect for {occasion} occasions.")
    elif style_rating < 5: feedback_parts.append(f"Not ideal for {occasion} events.")
    
    return main + " " + " ".join(feedback_parts) if feedback_parts else main

# --------------------------
# Routes
# --------------------------

@app.route("/")
def home():
    return jsonify({"status": "running", "models_loaded": True})

# --- UPDATED: /tryon Route (Handles URLs from Merchant + API Keys) ---
@app.route("/tryon", methods=["POST"])
def tryon():
    global LAST_RESULT_IMAGE, LAST_PERSON_IMAGE
    
    # Check if merchant is allowed
    if request.headers.get("X-API-KEY") and not require_apikey(request):
        return jsonify({"error": "Unauthorized. Invalid API Key."}), 401
    
    person_file = request.files.get("person")
    garment_file = request.files.get("garment")
    garment_url = request.form.get("garment_url") # NEW: Accept URL from merchant
    
    if not person_file or (not garment_file and not garment_url):
        return jsonify({"error": "Upload person image and provide garment file or URL"}), 400
    
    try:
        person_bytes = person_file.read()
        LAST_PERSON_IMAGE = base64.b64encode(person_bytes).decode('utf-8')
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        person_buf = io.BytesIO()
        person_img.save(person_buf, format="PNG")
        
        # Determine if we are reading a file upload (original dashboard) or a URL (merchant site)
        # Determine if we are reading a file upload (original dashboard) or a URL (merchant site)
        if garment_url:
            print(f"🔗 Downloading garment from URL: {garment_url}")
            
            # The Disguise: Pretend to be a normal Google Chrome browser
            stealth_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "image/webp,image/apng,image/*,*/*;q=0.8"
            }
            
            # Download the image using the disguise
            dl_response = requests.get(garment_url, headers=stealth_headers, timeout=10)
            
            # Check if the website still blocked us
            if dl_response.status_code != 200 or 'text/html' in dl_response.headers.get('Content-Type', ''):
                return jsonify({"error": "The merchant's server blocked the image download due to anti-bot protection. Please try a different garment."}), 400
                
            garment_bytes = dl_response.content
        else:
            garment_bytes = garment_file.read()
            
        garment_img = Image.open(io.BytesIO(garment_bytes)).convert("RGB")
        garment_buf = io.BytesIO()
        garment_img.save(garment_buf, format="PNG")
        
        headers = {
            "x-rapidapi-key": RAPIDAPI_KEY,
            "x-rapidapi-host": RAPIDAPI_HOST
        }
        
        files = {
            "image": ("person.png", person_buf.getvalue(), "image/png"),
            "image-apparel": ("garment.png", garment_buf.getvalue(), "image/png")
        }
        
        response = requests.post(API_ENDPOINT, headers=headers, files=files, timeout=120)
        print(f"✅ RapidAPI status: {response.status_code}")
        
        if response.status_code != 200:
            return jsonify({"error": response.text}), response.status_code
            
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            data = response.json()
            img_b64 = data["results"][0]["entities"][0]["image"]
        else:
            img_b64 = base64.b64encode(response.content).decode()
        
        LAST_RESULT_IMAGE = img_b64
        return jsonify({"image": img_b64})
        
    except Exception as e:
        print(f"❌ Try-on error: {e}")
        return jsonify({"error": str(e)}), 500

# --- UPDATED: /rate Route (Recommendations Removed to Prevent Clashes) ---
@app.route("/rate", methods=["POST"])
def rate():
    global LAST_RESULT_IMAGE, LAST_PERSON_IMAGE
    
    if LAST_RESULT_IMAGE is None:
        return jsonify({"error": "Generate try-on first"}), 400
    
    try:
        data = request.json
        occasion = data.get("occasion", "casual").lower()
        
        print("\n" + "="*60)
        print("🔍 ADVANCED FEATURE EXTRACTION")
        print("="*60)
        
        color = extract_dominant_color_kmeans(LAST_RESULT_IMAGE)
        print(f"✅ Detected color: {color}")
        
        skin = detect_skin_tone_advanced(LAST_PERSON_IMAGE) if LAST_PERSON_IMAGE else "medium"
        print(f"✅ Detected skin tone: {skin}")
        
        style = infer_style_from_occasion(occasion)
        
        features = encode_input_safe(occasion, color, skin, style)
        if features is None:
            return jsonify({"error": "Feature encoding failed"}), 500
        
        color_rating = float(color_model.predict(features)[0])
        style_rating = float(style_model.predict(features)[0])
        overall_rating = float(overall_model.predict(features)[0])
        
        comment = generate_comment(color_rating, style_rating, overall_rating, occasion)
        
        return jsonify({
            "color_rating": round(color_rating, 1),
            "style_rating": round(style_rating, 1),
            "overall_rating": round(overall_rating, 1),
            "comment": comment,
            "detected_features": {
                "occasion": occasion,
                "color": color,
                "skin_tone": skin,
                "style": style
            }
        })
        
    except Exception as e:
        print(f"❌ Rating error: {e}")
        return jsonify({"error": str(e)}), 500

# --- NEW: /recommend Route (Standalone for Merchant Website) ---
@app.route("/recommend", methods=["POST"])
def get_recommendations():
    """Headless Recommendation API - Returns Search Tags, Not Items!"""
    global LAST_PERSON_IMAGE
    
    # Security check
    if not require_apikey(request):
        return jsonify({"error": "Unauthorized"}), 401
        
    try:
        data = request.json
        
        # 1. Receive the Garment Metadata from the Merchant Website
        garment_age = data.get("age_group", "women").lower()
        garment_gender = data.get("gender", "female").lower()
        target_style = data.get("style", "casual").lower()
        target_color = data.get("color", "blue").lower()

        print("\n🛒 [HEADLESS API] Analyzing user for merchant tags...")
        
        # 2. AI Face Scan (What DeepFace thinks)
        detected_age_group, detected_gender = detect_demographic(LAST_PERSON_IMAGE, garment_age, garment_gender)
        
        print(f"🤖 AI Detected: {detected_age_group} ({detected_gender})")
        print(f"👗 Garment Is: {garment_age} ({garment_gender})")

        # 3. THE SMART OVERRIDE LOGIC (Upgraded for E-commerce)
        
        # RULE 1: NEVER change the age group. If they are browsing Kids/Boys/Baby, stay there!
        final_age = garment_age 
        
        # RULE 2: Handle Gender Mismatches
        if detected_gender != garment_gender and garment_gender != "unisex":
            print("⚠️ Gender Mismatch! Overriding AI. Trusting Garment Metadata.")
            final_gender = garment_gender
        elif garment_gender == "unisex":
            print(f"✅ Unisex item. Tailoring recommendations to AI detected gender: {detected_gender}")
            final_gender = detected_gender
        else:
            print("✅ Match Confirmed.")
            final_gender = garment_gender

        # 4. Generate the Smart Search Query to send BACK to the Merchant
        search_query = {
            "target_age_group": final_age,
            "target_gender": final_gender,
            "target_style": target_style,
            "target_color": target_color
        }
        
        print(f"📡 Sending Search Query back to Merchant: {search_query}")
        
        # Notice we are returning TAGS, not clothes!
        return jsonify({"search_query": search_query})

    except Exception as e:
        print(f"❌ Recommendation error: {e}")
        return jsonify({"error": str(e)}), 500

# --------------------------
# Run server
# --------------------------
if __name__ == "__main__":
    # 1. Get the port from Render's environment, default to 5000 for local testing
    port = int(os.environ.get("PORT", 5000))
    
    print("\n" + "="*60)
    print("🚀 VIRTUAL TRY-ON API SERVER (PRODUCTION EDITION)")
    print("="*60)
    print(f"✅ Running on Port: {port}")
    print("🔒 B2B API Key Security Active")
    print("🌐 Ready for Merchant Requests")
    print("="*60 + "\n")
    
    # 2. Bind to 0.0.0.0 so it's accessible from the internet
    # 3. Disable debug=True for production (it's a security risk)
    app.run(host="0.0.0.0", port=port, debug=False)



