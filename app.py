import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for
from torchvision import transforms
from PIL import Image
import timm
import re
import sqlite3

# ================= CONFIG =================
IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['Autumn', 'Monsoon', 'Spring', 'Summer', 'Winter']
CLINICAL_COLS = ['avg_temp','humidity','rainfall','sunlight_hours','wind_speed','leaf_wetness']

UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "Models/fusion_effnetb3.pth"
SCALER_PATH = "Models/envi_scaler.pkl"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= FLASK =================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ================= TRANSFORMS =================
img_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ================= BACKBONE =================
backbone = timm.create_model("efficientnet_b3", pretrained=False)
backbone.classifier = nn.Identity()
backbone.eval().to(DEVICE)

def backbone_features(x):
    feat_map = backbone.forward_features(x)
    if feat_map.ndim == 2:
        return feat_map
    pooled = backbone.global_pool(feat_map)
    return pooled.view(pooled.size(0), -1)

# ================= FUSION MODEL =================
class FusionModel(nn.Module):
    def __init__(self, img_feat_dim, clin_dim, num_classes=5):
        super().__init__()
        self.img_mlp = nn.Sequential(
            nn.Linear(img_feat_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.clin_mlp = nn.Sequential(
            nn.Linear(clin_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, img_feat, clin_feat):
        img_out = self.img_mlp(img_feat)
        clin_out = self.clin_mlp(clin_feat)
        fused = torch.cat([img_out, clin_out], dim=1)
        return self.classifier(fused)

# ================= LOAD MODEL =================
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# infer image feature size
with torch.no_grad():
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    IMG_FEAT_DIM = backbone_features(dummy).shape[1]

model = FusionModel(IMG_FEAT_DIM, len(CLINICAL_COLS), num_classes=5)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval().to(DEVICE)

backbone.load_state_dict(checkpoint['backbone_state_dict'])

# load scaler
scaler_clin = joblib.load(SCALER_PATH)

# ================= ROUTES =================

@app.route('/predict', methods=['POST'])
def predict():

    # ----- image -----
    img_file = request.files['image']
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
    img_file.save(img_path)

    img = Image.open(img_path).convert('RGB')
    img = img_tfms(img).unsqueeze(0).to(DEVICE)

    # ----- clinical inputs -----
    clin_vals = [float(request.form[col]) for col in CLINICAL_COLS]
    clin_arr = np.array(clin_vals).reshape(1, -1)
    clin_scaled = scaler_clin.transform(clin_arr)
    clin_tensor = torch.tensor(clin_scaled, dtype=torch.float32).to(DEVICE)

    # ----- inference -----
    with torch.no_grad():
        img_feat = backbone_features(img)
        logits = model(img_feat, clin_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)

    result = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100

    prob_dict = {CLASS_NAMES[i]: f"{probs[i]*100:.2f}%" for i in range(len(CLASS_NAMES))}

    return render_template(
        'result.html',
        result=result,
        confidence=f"{confidence:.2f}%",
        probs=prob_dict,
        image_path=img_path
    )


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")
    else:
        username = request.form.get('user','')
        name = request.form.get('name','')
        email = request.form.get('email','')
        number = request.form.get('mobile','')
        password = request.form.get('password','')

        # Server-side validation
        username_pattern = r'^.{6,}$'
        name_pattern = r'^[A-Za-z ]{3,}$'
        email_pattern = r'^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$'
        mobile_pattern = r'^[6-9][0-9]{9}$'
        password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$'

        if not re.match(username_pattern, username):
            return render_template("signup.html", message="Username must be at least 6 characters.")
        if not re.match(name_pattern, name):
            return render_template("signup.html", message="Full Name must be at least 3 letters, only letters and spaces allowed.")
        if not re.match(email_pattern, email):
            return render_template("signup.html", message="Enter a valid email address.")
        if not re.match(mobile_pattern, number):
            return render_template("signup.html", message="Mobile must start with 6-9 and be 10 digits.")
        if not re.match(password_pattern, password):
            return render_template("signup.html", message="Password must be at least 8 characters, with an uppercase letter, a number, and a lowercase letter.")

        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("SELECT 1 FROM info WHERE user = ?", (username,))
        if cur.fetchone():
            con.close()
            return render_template("signup.html", message="Username already exists. Please choose another.")
        
        cur.execute("insert into `info` (`user`,`name`, `email`,`mobile`,`password`) VALUES (?, ?, ?, ?, ?)",(username,name,email,number,password))
        con.commit()
        con.close()
        return redirect(url_for('login'))

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "GET":
        return render_template("signin.html")
    else:
        mail1 = request.form.get('user','')
        password1 = request.form.get('password','')
        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
        data = cur.fetchone()

        if data == None:
            return render_template("signin.html", message="Invalid username or password.")    

        elif mail1 == 'admin' and password1 == 'admin':
            return render_template("home.html")

        elif mail1 == str(data[0]) and password1 == str(data[1]):
            return render_template("home.html")
        else:
            return render_template("signin.html", message="Invalid username or password.")

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/home1')
def home1():
	return render_template('home1.html')

@app.route("/graphs")
def graphs():
    return render_template("graphs.html")

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')


# ================= RUN =================
if __name__ == '__main__':
    app.run(debug=False)
