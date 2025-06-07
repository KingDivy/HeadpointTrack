from flask import Flask, render_template, Response
import cv2 as cv
import mediapipe as mp

app = Flask(__name__)

cap = cv.VideoCapture(0)

fmesh = mp.solutions.face_mesh
facemesh = fmesh.FaceMesh(max_num_faces=3, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def generate_frames():
    while True:
        success, vid = cap.read()

        vidRGB = cv.cvtColor(vid, cv.COLOR_BGR2RGB)
        results = facemesh.process(vidRGB)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmark = face_landmarks.landmark[151]
                h, w, _ = vid.shape
                x, y = int(landmark.x * w), int(landmark.y * h)

                cv.circle(vid, (x, y), 8, (255, 255, 0), -1)
                cv.putText(vid, f"({x}, {y})", (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        ret, buffer = cv.imencode('.jpg', vid)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)