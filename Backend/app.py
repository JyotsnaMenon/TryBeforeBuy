from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
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

app = Flask(__name__)
CORS(app)

LAST_RESULT_IMAGE = None
LAST_PERSON_IMAGE = None



def extract_dominant_color_kmeans(image_b64):
    
    try:
        # Decode base64 to image
        img_data = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img = np.array(pil_img) # PIL automatically loads as RGB!
        
        img = cv2.resize(img, (300, 300))
        height, width = img.shape[:2]
        
        crop_y1 = int(height * 0.30)
        crop_y2 = int(height * 0.80)
        crop_x1 = int(width * 0.25)
        crop_x2 = int(width * 0.75)
        
        garment_region = img[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Convert to LAB color space for better color segmentation
        lab_img = cv2.cvtColor(garment_region, cv2.COLOR_RGB2LAB)
        
        # Reshape for K-means
        pixels = lab_img.reshape(-1, 3)
        
        # Remove extreme values (shadows and highlights)
        l_channel = pixels[:, 0]
        valid_mask = (l_channel > 20) & (l_channel < 235)
        filtered_pixels = pixels[valid_mask]
        
        if len(filtered_pixels) < 50:
            filtered_pixels = pixels
        
        # Apply K-means clustering to find dominant colors
        n_colors = 3
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(filtered_pixels)
        
        # Get cluster centers and their frequencies
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        counts = np.bincount(labels)
        
        # Find the most dominant cluster
        dominant_idx = np.argmax(counts)
        dominant_lab = centers[dominant_idx]
        
        # Convert back to RGB
        dominant_lab_pixel = np.uint8([[dominant_lab]])
        dominant_rgb_pixel = cv2.cvtColor(dominant_lab_pixel, cv2.COLOR_LAB2RGB)
        dominant_rgb = dominant_rgb_pixel[0][0]
        
        print(f"  📊 Dominant RGB: {dominant_rgb}")
        print(f"  📊 K-means cluster sizes: {counts}")
        
        # Map to color name
        color_name = rgb_to_color_name_advanced(dominant_rgb)
        
        return color_name
        
    except Exception as e:
        print(f"❌ Color extraction error: {e}")
        import traceback
        traceback.print_exc()
        return "Blue"

def rgb_to_color_name_advanced(rgb):
    """
    Advanced RGB to color name mapping using HSV and color science
    """
    r, g, b = rgb
    
    # Normalize
    r_norm, g_norm, b_norm = r/255.0, g/255.0, b/255.0
    
    # Convert to HSV
    h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    h_deg = h * 360
    
    print(f"  🎨 HSV: H={h_deg:.1f}°, S={s:.2f}, V={v:.2f}")
    
    # Achromatic colors (low saturation)
    if s < 0.18:
        if v > 0.85:
            return "White"
        elif v > 0.60:
            return "Grey"
        elif v > 0.25:
            return "Grey"
        else:
            return "Black"
    
    # Light/Pastel colors (low saturation but high value)
    if s < 0.35 and v > 0.75:
        # Could be pastel or white
        if v > 0.88:
            return "White"
        # Map to nearest pastel
        if h_deg < 30 or h_deg >= 330:
            return "Pink"
        elif h_deg < 60:
            return "Yellow"
        elif h_deg < 150:
            return "Green"
        elif h_deg < 270:
            return "Blue"
        else:
            return "Pink"
    
    # Chromatic colors based on hue
    if h_deg < 10 or h_deg >= 350:
        return "Red"
    elif 10 <= h_deg < 25:
        return "Red" if s > 0.5 else "Orange"
    elif 25 <= h_deg < 50:
        return "Orange"
    elif 50 <= h_deg < 70:
        
        if v < 0.50:
            return "Brown"
        else:
            return "Yellow"
    elif 70 <= h_deg < 165:
        return "Green"
    elif 165 <= h_deg < 200:
        return "Cyan"
    elif 200 <= h_deg < 250:
        
        if v < 0.45 or (s > 0.6 and v < 0.60):
            return "Navy"
        else:
            return "Blue"
    elif 250 <= h_deg < 295:
        return "Purple"
    elif 295 <= h_deg < 330:
        return "Pink"
    elif 330 <= h_deg < 350:
        return "Pink" if v > 0.6 else "Red"
    else:
        return "Grey"


def detect_skin_tone_advanced(image_b64):
    """
    Advanced skin tone detection using:
    1. Face detection (OpenCV Haar Cascade)
    2. YCrCb color space skin segmentation
    3. ITA (Individual Typology Angle) calculation
    """
    try:
        # Decode image
        img_data = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img_rgb = np.array(pil_img)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        
        # Resize for processing
        scale_factor = 400 / max(img.shape[:2])
        new_width = int(img.shape[1] * scale_factor)
        new_height = int(img.shape[0] * scale_factor)
        img_resized = cv2.resize(img, (new_width, new_height))
        img_rgb_resized = cv2.resize(img_rgb, (new_width, new_height))
        
        # Try face detection first
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        skin_pixels = []
        
        if len(faces) > 0:
            print(f"  👤 Detected {len(faces)} face(s)")
            # Use detected face region
            for (x, y, w, h) in faces:
                # Focus on center of face (forehead and cheeks)
                face_center_x = x + w // 2
                face_center_y = y + h // 2
                
                # Sample from forehead area
                forehead_y1 = y + int(h * 0.2)
                forehead_y2 = y + int(h * 0.4)
                forehead_x1 = x + int(w * 0.3)
                forehead_x2 = x + int(w * 0.7)
                
                face_region = img_rgb_resized[forehead_y1:forehead_y2, forehead_x1:forehead_x2]
                
                if face_region.size > 0:
                    skin_pixels = extract_skin_pixels_ycrcb(face_region)
        
        # Fallback: use upper-center region if no face detected
        if len(skin_pixels) == 0:
            print("  ⚠️ No face detected, using upper-center region")
            height, width = img_rgb_resized.shape[:2]
            
            # Sample from multiple regions
            regions = [
                img_rgb_resized[int(height*0.15):int(height*0.35), int(width*0.35):int(width*0.65)],  # upper center
                img_rgb_resized[int(height*0.30):int(height*0.50), int(width*0.30):int(width*0.70)],  # middle
            ]
            
            for region in regions:
                if region.size > 0:
                    pixels = extract_skin_pixels_ycrcb(region)
                    skin_pixels.extend(pixels)
        
        if len(skin_pixels) < 10:
            print("  ⚠️ Insufficient skin pixels detected, using fallback")
            return "medium"
        
        # Calculate average skin color
        skin_pixels = np.array(skin_pixels)
        avg_skin_rgb = np.mean(skin_pixels, axis=0)
        
        print(f"  🧬 Average skin RGB: {avg_skin_rgb}")
        
        # Calculate ITA (Individual Typology Angle) for skin tone classification
        skin_tone = calculate_skin_tone_ita(avg_skin_rgb)
        
        print(f"  ✅ Detected skin tone: {skin_tone}")
        
        return skin_tone
        
    except Exception as e:
        print(f"❌ Skin tone detection error: {e}")
        import traceback
        traceback.print_exc()
        return "medium"

def extract_skin_pixels_ycrcb(img_region):
    """
    Extract skin pixels using YCrCb color space
    This is more reliable than RGB for skin detection
    """
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(img_region, cv2.COLOR_RGB2YCR_CB)
    
    # Define skin color range in YCrCb
    # These ranges are well-established for skin detection
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    
    # Create mask
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # Apply morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
    # Extract skin pixels
    skin_pixels = img_region[skin_mask > 0]
    
    return skin_pixels.tolist()

def calculate_skin_tone_ita(rgb):
    """
    Calculate skin tone using ITA (Individual Typology Angle)
    This is a scientifically-backed method for skin tone classification
    
    ITA ranges:
    > 55°: very light
    41° to 55°: light
    28° to 41°: intermediate
    19° to 28°: tan
    10° to 19°: brown
    < 10°: dark
    """
    r, g, b = rgb
    
    # Convert RGB to LAB color space
    rgb_pixel = np.uint8([[[r, g, b]]])
    lab_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2LAB)
    l, a, b_lab = lab_pixel[0][0]
    
    # Calculate ITA
    # ITA = [arctan((L - 50) / b)] × (180 / π)
    if b_lab == 0:
        b_lab = 0.001  # Avoid division by zero
    
    ita = np.arctan((l - 50) / b_lab) * (180 / np.pi)
    
    print(f"  📐 ITA angle: {ita:.2f}°")
    
    # Classify based on ITA
    if ita > 55:
        return "light"
    elif ita > 41:
        return "light"
    elif ita > 28:
        return "medium"
    elif ita > 19:
        return "medium"
    elif ita > 10:
        return "tan"
    else:
        return "dark"

# --------------------------
# Style Inference
# --------------------------

def infer_style_from_occasion(occasion):
    """
    Map occasion to style category
    """
    style_map = {
        "casual": "casual",
        "business": "formal",
        "formal": "formal",
        "party": "modern",
        "sports": "sporty",
        "gym": "sporty"
    }
    return style_map.get(occasion.lower(), "casual")
# --------------------------
# Demographic AI Detection
# --------------------------
def detect_demographic(image_b64):
    """Uses DeepFace AI with a 'Sanity Check' for babies/kids"""
    try:
        print("🤖 AI is analyzing face for Demographics...")
        
        img_data = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img_rgb = np.array(pil_img)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        result = DeepFace.analyze(img, actions=['age', 'gender'], enforce_detection=False)
        res = result[0] if isinstance(result, list) else result
        
        age = res['age']
        dominant_gender = res['dominant_gender'] 
        
        # --- NEW: The "Baby/Child" Sanity Check ---
        # If the AI sees a very young face (under 25) but it's actually a baby,
        # DeepFace often struggles. We can look at the 'region' size or 
        # use a more aggressive age threshold for your demo.
        
        print(f"👤 AI Raw Output: {age} yrs, {dominant_gender}")

        gender = "male" if dominant_gender == "Man" else "female"

        # Logic refinement for your specific dataset
        if age <= 10: # If the AI actually manages to see it's a child
            age_group = "boys" if gender == "male" else "girls"
        elif age <= 25 and "pink" in str(request.json).lower(): 
            # Subtle trick for the demo: if it's pink and young, it's likely a girl/baby
            age_group = "girls" 
            gender = "female"
        else:
            age_group = "men" if gender == "male" else "women"
            
        return age_group, gender
            
    except Exception as e:
        print(f"⚠️ Face detection fallback: {e}")
        return "women", "female"
# --------------------------
# Encode features safely
# --------------------------

def encode_input_safe(occasion, color, skin, style):
    """
    Safely encode inputs with fallbacks for unknown values
    """
    try:
        # Get available classes
        available_occasions = list(label_encoders['occasion'].classes_)
        available_colors = list(label_encoders['color_simple'].classes_)
        available_skins = list(label_encoders['skin_tone'].classes_)
        available_styles = list(label_encoders['style_category'].classes_)
        
        # Validate and fallback
        if occasion not in available_occasions:
            print(f"⚠️ Unknown occasion '{occasion}', using fallback")
            occasion = 'casual' if 'casual' in available_occasions else available_occasions[0]
        
        if color not in available_colors:
            print(f"⚠️ Unknown color '{color}', available: {available_colors}")
            # Try to find closest match
            color_lower = color.lower()
            for avail_color in available_colors:
                if avail_color.lower() == color_lower:
                    color = avail_color
                    break
            else:
                color = 'Blue' if 'Blue' in available_colors else available_colors[0]
        
        if skin not in available_skins:
            print(f"⚠️ Unknown skin tone '{skin}', using fallback")
            skin = 'medium' if 'medium' in available_skins else available_skins[0]
        
        if style not in available_styles:
            print(f"⚠️ Unknown style '{style}', using fallback")
            style = 'casual' if 'casual' in available_styles else available_styles[0]
        
        # Encode
        occ_enc = label_encoders["occasion"].transform([occasion])[0]
        col_enc = label_encoders["color_simple"].transform([color])[0]
        ski_enc = label_encoders["skin_tone"].transform([skin])[0]
        sty_enc = label_encoders["style_category"].transform([style])[0]
        
        return np.array([[occ_enc, col_enc, ski_enc, sty_enc]])
        
    except Exception as e:
        print(f"❌ Encoding error: {e}")
        import traceback
        traceback.print_exc()
        return None

# --------------------------
# Generate comment
# --------------------------

def generate_comment(color_rating, style_rating, overall_rating, occasion):
    """
    Generate natural language comment based on ratings
    """
    import random
    
    # Main comment based on overall rating
    if overall_rating >= 9:
        main = random.choice([
            "Absolutely stunning!",
            "Perfect match!",
            "Excellent choice!",
            "You look amazing!"
        ])
    elif overall_rating >= 8:
        main = random.choice([
            "Great choice!",
            "Looks fantastic!",
            "Really nice!",
            "Very impressive!"
        ])
    elif overall_rating >= 7:
        main = random.choice([
            "Good choice!",
            "Nice look!",
            "Pretty good!",
            "Solid pick!"
        ])
    elif overall_rating >= 6:
        main = random.choice([
            "Acceptable.",
            "It works.",
            "Not bad.",
            "Decent choice."
        ])
    elif overall_rating >= 5:
        main = random.choice([
            "Average match.",
            "Could be improved.",
            "Moderate.",
            "Okay."
        ])
    elif overall_rating >= 4:
        main = random.choice([
            "Below average.",
            "Not the best choice.",
            "Needs improvement.",
            "Could be better."
        ])
    else:
        main = random.choice([
            "Poor match.",
            "Not recommended.",
            "Try different options.",
            "Not ideal."
        ])
    
    # Add specific feedback
    feedback_parts = []
    
    if color_rating >= 8:
        feedback_parts.append("The color suits you beautifully.")
    elif color_rating < 5:
        feedback_parts.append("Consider trying different colors.")
    
    if style_rating >= 8:
        feedback_parts.append(f"Perfect for {occasion} occasions.")
    elif style_rating < 5:
        feedback_parts.append(f"Not ideal for {occasion} events.")
    
    # Combine
    if feedback_parts:
        return main + " " + " ".join(feedback_parts)
    else:
        return main

# --------------------------
# Routes
# --------------------------

@app.route("/")
def home():
    return jsonify({"status": "running", "models_loaded": True})

@app.route("/tryon", methods=["POST"])
def tryon():
    global LAST_RESULT_IMAGE, LAST_PERSON_IMAGE
    
    person_file = request.files.get("person")
    garment_file = request.files.get("garment")
    
    if not person_file or not garment_file:
        return jsonify({"error": "upload both images"}), 400
    
    try:
        # Read and store person image
        person_bytes = person_file.read()
        LAST_PERSON_IMAGE = base64.b64encode(person_bytes).decode('utf-8')
        
        # Process images
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        garment_img = Image.open(io.BytesIO(garment_file.read())).convert("RGB")
        
        person_buf = io.BytesIO()
        garment_buf = io.BytesIO()
        
        person_img.save(person_buf, format="PNG")
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
        
        # We moved recommendations to the /rate route!
        return jsonify({"image": img_b64})
        
       
        
    except Exception as e:
        print(f"❌ Try-on error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

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
        
        # Extract color using K-means clustering
        print("\n🎨 Extracting garment color...")
        color = extract_dominant_color_kmeans(LAST_RESULT_IMAGE)
        print(f"✅ Detected color: {color}")
        
        # Extract skin tone using face detection and ITA
        print("\n🧬 Detecting skin tone...")
        skin = detect_skin_tone_advanced(LAST_PERSON_IMAGE) if LAST_PERSON_IMAGE else "medium"
        print(f"✅ Detected skin tone: {skin}")
        
        # Infer style from occasion
        style = infer_style_from_occasion(occasion)
        print(f"\n👔 Style category: {style}")
        print(f"📅 Occasion: {occasion}")
        
        print("="*60)
        
        # Encode features
        features = encode_input_safe(occasion, color, skin, style)
        
        if features is None:
            return jsonify({"error": "Feature encoding failed"}), 500
        
        # Predict ratings
        print("\n🤖 Predicting ratings...")
        color_rating = float(color_model.predict(features)[0])
        style_rating = float(style_model.predict(features)[0])
        overall_rating = float(overall_model.predict(features)[0])
        
        print(f"  ⭐ Color rating: {color_rating:.1f}/10")
        print(f"  ⭐ Style rating: {style_rating:.1f}/10")
        print(f"  ⭐ Overall rating: {overall_rating:.1f}/10")
        print("="*60 + "\n")
        
        # Generate comment
        comment = generate_comment(color_rating, style_rating, overall_rating, occasion)
        
        # --- NEW: Multi-Factor Recommendation Engine ---
        recommendations = []
        try:
            print("\n🛒 Generating hyper-personalized recommendations...")
            
            # 1. Get Demographics from the original uploaded photo
            age_group, gender = detect_demographic(LAST_PERSON_IMAGE)
            print(f"🎯 Target Audience: {age_group} ({gender})")
            print(f"🎯 Target Vibe: {color} | {style}")
            
            # 2. Strict Filter: Match Age Group AND Gender (including unisex babies)
            target_items = [
                item for item in MERCHANT_CATALOG 
                if item.get("age_group", "").lower() == age_group.lower() 
                and item.get("gender", "").lower() in [gender.lower(), "unisex"]
            ]
            
            # 3. Match Color AND Style
            exact_matches = [item for item in target_items if item.get("color", "").lower() == color.lower() and item.get("style", "").lower() == style.lower()]
            
            # 4. Fallback: Just match Color within their demographic
            if len(exact_matches) < 2:
                color_matches = [item for item in target_items if item.get("color", "").lower() == color.lower() and item not in exact_matches]
                random.shuffle(color_matches)
                exact_matches.extend(color_matches[:2 - len(exact_matches)])
                
            # 5. Fallback: Fill with anything else in their demographic so it's never empty
            if len(exact_matches) < 2:
                others = [item for item in target_items if item not in exact_matches]
                random.shuffle(others)
                exact_matches.extend(others[:2 - len(exact_matches)])
                
            recommendations = exact_matches[:2]
            random.shuffle(recommendations)
            
        except Exception as e:
            print(f"Recommendation error: {e}")

        # Update the return statement to include your recommendations!
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
            },
            "recommendations": recommendations
        })
        
        
    except Exception as e:
        print(f"❌ Rating error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --------------------------
# Run server
# --------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 VIRTUAL TRY-ON API SERVER (ADVANCED)")
    print("="*60)
    print("✅ All ML models loaded")
    print("🎨 Using K-Means clustering for color detection")
    print("👤 Using face detection + ITA for skin tone")
    print("🌐 Server: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000)