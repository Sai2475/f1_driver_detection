# 🏁 GridScan: F1 Driver Detector

GridScan is an AI-powered web application that identifies Formula 1 drivers from images using facial recognition and large language models. Upload an image of an F1 driver—past or present—and the app will return their full profile, including their team, car number, championship wins, and current status (active, retired, or deceased).

## 🚀 Features

- 🎯 Detects the most prominent F1 driver's face from an image.
- 🧠 Uses LLaMA 4 via Groq API for intelligent driver identification.
- 📸 Displays driver name, nationality, team, car number, and status.
- ⚰️ Indicates if the driver is retired or deceased (with year).
- 💻 Modern UI built with Flask, Tailwind CSS & OpenCV.

---

## 🧪 Tech Stack

- **Frontend**: HTML5, Tailwind CSS
- **Backend**: Python, Flask
- **ML Services**: OpenCV (face detection), Groq + LLaMA 4 (driver profiling)
- **Others**: RESTful APIs, Jinja2 templates, base64 image encoding

