from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import base64
import requests

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

GROQ_API_KEY = ""  
GROQ_API_URL = ""

def detect_faces(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        return None, img_rgb

    largest_face = max(faces, key=lambda r: r[2] * r[3])
    return [largest_face], img_rgb

def draw_faces(image, faces, player_name):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        font_scale = max(0.5, min(1.5, 30 / len(player_name)))
        cv2.putText(image, player_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    return image

def get_player_info(image_path):
    with open(image_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode()

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
You are a highly intelligent AI specialized in identifying professional Formula 1 drivers from photographs. Carefully analyze the uploaded image and match the visible face to an F1 driver who has competed in the Formula 1 World Championship at any point in history.

Respond with the following structured information if a well-known F1 driver is identified:

- **Full Name**: (Driver's complete name)
- **Nationality**: (Country they race(d) under)
- **Team**: (Most recent or most iconic team)
- **Car Number**: (Current or most recognized racing number)
- **Championships**: (Notable achievements — World Titles, Wins, or Records — max 3-4)

If multiple people are in the image, return only the **most famous driver**.

If the person in the image is **not** an F1 driver or the face is not clearly visible, respond with:
**"Unknown"**.
"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        return "Unknown"

@app.route("/", methods=["GET", "POST"])
def index():
    player_info = ""
    result_img_filename = ""

    if request.method == "POST":
        file = request.files['image']
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            faces, img_rgb = detect_faces(image_path)

            if faces:
                player_info = get_player_info(image_path)
                player_name = player_info.split("\n")[0].replace("**Full Name**: ", "").strip()
                processed_img = draw_faces(img_rgb, faces, player_name)

                result_img_filename = f"result_{file.filename}"
                result_img_path = os.path.join(app.config['UPLOAD_FOLDER'], result_img_filename)
                cv2.imwrite(result_img_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
            else:
                player_info = "No faces detected. Try another image."

    return render_template("index.html", player_info=player_info, result_img=result_img_filename)

if __name__ == "__main__":
    app.run(debug=True)
