# app.py
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from ultralytics import YOLO
import cv2

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

model = YOLO("yolov8n.pt")
caps = cv2.VideoCapture("intersection.mp4")

# Define lane rectangles (manually tune these)
LANES = {
    'North-South': (200, 0, 400, 480),
    'East-West': (0, 240, 800, 480)
}

counts = {l:0 for l in LANES}
emergency = False

def detect_loop():
    global counts, emergency
    for res in model.track(source="intersection.mp4", tracker="bytetrack.yaml", stream=True):
        frame = res.orig_img
        counts = {l:0 for l in LANES}
        emergency = False

        for box, cls_id, _ in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.id):
            x1,y1,x2,y2 = map(int, box)
            cx,cy = (x1+x2)//2,(y1+y2)//2
            for lane,(ax,ay,bx,by) in LANES.items():
                if ax<cx<bx and ay<cy<by:
                    counts[lane] += 1
            if int(cls_id)==9:  # ambulance
                emergency = True

        # send updates
        socketio.emit("update", {"counts":counts, "emergency": emergency})
        socketio.sleep(0.1)

@app.route("/")
def index():
    return render_template("index.html")

def gen():
    while caps.isOpened():
        ret, frame = caps.read()
        if not ret: break
        # draw lanes and counts
        for l,(ax,ay,bx,by) in LANES.items():
            cv2.rectangle(frame,(ax,ay),(bx,by),(255,0,0),2)
            cv2.putText(frame, f"{l}:{counts[l]}", (ax,ay-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0),2)
        if emergency:
            cv2.putText(frame,"🚨 EMERGENCY",(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        _, jpg = cv2.imencode('.jpg',frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpg.tobytes()+b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__=="__main__":
    socketio.start_background_task(detect_loop)
    socketio.run(app, host="0.0.0.0", port=5000)
