from flask import Flask, request, jsonify, send_from_directory # ⭐️ 1. NAYA CHANGE
from flask_cors import CORS
from PIL import Image
import requests
from io import BytesIO
import torch
from torchvision import transforms, models
import os
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import base64
import uuid 

app = Flask(__name__)
CORS(app)

# ---------- Database connection ----------
def get_db():
    return mysql.connector.connect(
        host="localhost", 
        user="root",
        password="Muskan@1707", 
        database="waste_ai"
    )

# ---------- ML Model Setup ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "waste_classifier.pth")
FRONTEND_FOLDER = os.path.join(BASE_DIR, "frontend") # ⭐️ 2. NAYA CHANGE

model = models.resnet50(pretrained=False) 
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4) 

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

class_names = ["general", "hazardous", "organic", "recyclable"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


#

@app.route("/report-dump", methods=["POST"])
def report_dump():
    try:
       
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        user_id = request.form.get('user_id')
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        notes = request.form.get('notes', '') # Notes optional hai

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not user_id or not latitude or not longitude:
            return jsonify({"error": "Missing user_id or location data"}), 400
        
        # ⭐️ File ko save karein (Ek naya folder 'reports' banakar)
        # (Hum BASE_DIR ka istemal karenge jo pehle se defined hai)
        REPORTS_DIR = os.path.join(BASE_DIR, "data", "reports")
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        # File ka naam unique banayein
        ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{ext}"
        image_path = os.path.join(REPORTS_DIR, unique_filename)
        
        file.save(image_path)
        
        # ⭐️ Database mein report save karein
        # (Iske liye aapko 'reports' table banani hogi)
        db = get_db()
        cursor = db.cursor()
        
        
        cursor.execute(
            """INSERT INTO reports 
            (user_id, latitude, longitude, image_filename, notes, timestamp) 
            VALUES (%s, %s, %s, %s, %s, %s)""",
            (user_id, latitude, longitude, unique_filename, notes, datetime.now())
        )
        db.commit()
        cursor.close()
        db.close()

        return jsonify({"message": "Report submitted successfully"})

    except Exception as e:
        print(f"Error in /report-dump: {e}")
        return jsonify({"error": str(e)}), 500
@app.route("/")
def index():
    return send_from_directory(FRONTEND_FOLDER, "index.html")

# Yeh route aapki baaki files (login.html, register.html, style.css) ko serve karega
@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(FRONTEND_FOLDER, filename)

# ⭐️⭐️⭐️ NAYA CHANGE KHATAM ⭐️⭐️⭐️


# ---------- ML Prediction Route ----------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        img = None
        filename = None
        user_id = 1 

        if "file" in request.files:
            file = request.files["file"]
            img = Image.open(file.stream).convert("RGB")
            filename = file.filename
            if "user_id" in request.form:
                user_id = int(request.form["user_id"])

        elif request.is_json:
            data = request.json
            if "user_id" in data:
                user_id = int(data["user_id"])

            if "url" in data:
                url = data["url"]
                if url.startswith("data:image/"):
                    header, encoded = url.split(",", 1)
                    img_bytes = base64.b64decode(encoded)
                    img = Image.open(BytesIO(img_bytes)).convert("RGB")
                    filename = "image_from_base64.jpg"
                elif url.startswith("http://") or url.startswith("https://"):
                    response = requests.get(url)
                    response.raise_for_status()
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    filename = url.split("/")[-1]
                else:
                    return jsonify({"error": "Invalid URL format"}), 400
            else:
                return jsonify({"error": "No file or URL provided"}), 400
        else:
            return jsonify({"error": "No file or URL provided"}), 400

        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(tensor)
            _, pred = torch.max(outputs, 1)
            predicted_class = class_names[pred.item()]

        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO predictions (user_id, filename, prediction, timestamp) VALUES (%s, %s, %s, %s)",
            (user_id, filename, predicted_class, datetime.now())
        )
        db.commit()
        cursor.close()
        db.close()

        return jsonify({"prediction": predicted_class})

    except Exception as e:
        print(f"Error in /predict: {e}") 
        return jsonify({"error": str(e)}), 500

# ---------- Auth & History Routes ----------
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    name = data.get("name")
    email = data.get("email") 
    password = data.get("password")

    if not email or not password or not name:
        return jsonify({"error": "Missing name, email, or password"}), 400

    db = get_db()
    cur = db.cursor(dictionary=True)
    
    cur.execute("SELECT * FROM users WHERE email=%s", (email,)) 
    if cur.fetchone():
        cur.close(); db.close()
        return jsonify({"error": "User already exists"}), 400

    hashed_pw = generate_password_hash(password)
    
    cur.execute(
        "INSERT INTO users (name, username, email, password_hash) VALUES (%s, %s, %s, %s)", 
        (name, email, email, hashed_pw)
    )
    db.commit(); cur.close(); db.close()
    return jsonify({"message": "User registered"}), 201

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email") 
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Missing email or password"}), 400

    db = get_db()
    cur = db.cursor(dictionary=True)
    cur.execute("SELECT * FROM users WHERE email=%s", (email,))
    user = cur.fetchone(); cur.close(); db.close()

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid credentials"}), 401

    return jsonify({
        "message": "Login successful", 
        "user_id": user["id"],
        "name": user.get("name") 
    })
@app.route("/history/<int:user_id>")
def history(user_id):
    db = get_db()
    cur = db.cursor(dictionary=True)
    cur.execute("SELECT id, filename, prediction, timestamp FROM predictions WHERE user_id=%s ORDER BY timestamp DESC", (user_id,))
    rows = cur.fetchall(); cur.close(); db.close()

    history_list = []
    for row in rows:
        history_list.append({
            "id": row["id"],
            "filename": row["filename"],
            "prediction": row["prediction"],
            "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if row["timestamp"] else None
        })

    return jsonify(history_list)

# ---------- Naye Routes (Gaadi Feature) ----------

@app.route("/areas", methods=["GET"])
def get_areas():
    try:
        db = get_db()
        cur = db.cursor(dictionary=True)
        cur.execute("SELECT area_name FROM area_schedules ORDER BY area_name")
        areas = [row['area_name'] for row in cur.fetchall()]
        cur.close(); db.close()
        return jsonify(areas)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/settings/<int:user_id>", methods=["GET", "POST"])
def user_settings(user_id):
    db = get_db()
    cur = db.cursor(dictionary=True)
    
    if request.method == "POST":
        data = request.json
        name = data.get("name")
        area = data.get("area") 

        cur.execute(
            "UPDATE users SET name = %s, area = %s WHERE id = %s",
            (name, area, user_id)
        )
        db.commit()
        cur.close(); db.close()
        return jsonify({"message": "Settings saved successfully"})
        
    else:
        cur.execute("SELECT name, email, area FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        cur.close(); db.close()
        if user:
            return jsonify({
                "name": user.get("name"),
                "email": user.get("email"), 
                "area": user.get("area")
            })
        else:
            return jsonify({"error": "User not found"}), 404

@app.route("/schedule/<int:user_id>", methods=["GET"])
def get_schedule(user_id):
    try:
        db = get_db()
        cur = db.cursor(dictionary=True)
        
        cur.execute("SELECT area FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        
        if not user or not user.get("area"):
            cur.close(); db.close()
            # ⭐️ Yeh error bhi English mein kar diya hai
            return jsonify({"error": "User has not set their area."}), 404
        
        user_area = user["area"]
        
        cur.execute("SELECT pickup_time FROM area_schedules WHERE area_name = %s", (user_area,))
        schedule = cur.fetchone()
        
        cur.close(); db.close()
        
        if schedule:
            return jsonify({
                "area": user_area,
                "pickup_time": schedule["pickup_time"]
            })
        else:
            return jsonify({"error": f"No schedule found for your area '{user_area}'."}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- Main ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)