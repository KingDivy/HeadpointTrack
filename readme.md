🎯 HeadShot Streaming
A real-time web application that streams your webcam feed and overlays facial landmarks using MediaPipe, OpenCV, and Flask.

<p align="center"> <img src="https://img.shields.io/badge/Python-3.9.13-blue?logo=python"> <img src="https://img.shields.io/badge/Flask-Web%20Framework-000000?logo=flask"> <img src="https://img.shields.io/badge/OpenCV-Image%20Processing-green?logo=opencv"> <img src="https://img.shields.io/badge/MediaPipe-Face%20Mesh-ff69b4?logo=google"> </p>
🔥 Features
📹 Live webcam video streaming in the browser

🧠 Real-time face mesh detection using MediaPipe

✨ Facial landmark visualization

🛠️ Built using:

Python 3.9.13

Flask

OpenCV

MediaPipe

📦 Requirements
Python 3.9.13

Flask

OpenCV (opencv-python)

MediaPipe

🚀 Installation
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/KingDivy/HeadpointTrack.git
cd HeadpointTrack
2. Create and activate a virtual environment (recommended)
bash
Copy
Edit
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
bash
Copy
Edit
pip install flask opencv-python mediapipe
🧪 Usage
Run the Flask app
bash
Copy
Edit
python headshot.py
Open in browser
cpp
Copy
Edit
http://127.0.0.1:5000/
You will see a real-time webcam feed with facial landmarks overlaid.

🗂️ Project Structure
php
Copy
Edit
HeadpointTrack/
├── static/
│   └── custom.css           # (Optional) Custom styling
├── templates/
│   └── index.html           # Frontend HTML page
├── headshot.py              # Main Flask application
├── requirements.txt         # Python dependencies
└── README.md                # Project info and usage
🤝 Contributions
Feel free to fork this repo, improve it, and submit a pull request. Suggestions and improvements are welcome!

📄 License
This project is licensed under the MIT License.
