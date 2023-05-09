from flask import Flask, render_template, Response
import cv2
import boto3
import time

app = Flask(__name__)

cam = cv2.VideoCapture(0)

ACCESS_KEY = "your_access_key"
SECRET_KEY = "your_secret_key"
BUCKET_NAME = "your_bucket_name"
FILE_NAME = "webcam_stream.mp4"

s3 = boto3.client("s3", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)


def upload_s3(frame):
    s3.upload_fileobj(frame, BUCKET_NAME, FILE_NAME)


@app.route("/")
def index():
    return render_template("index.html")


def gen(fps, recording_time):
    start_time = time.time()
    interval = 1 / fps
    while time.time() - start_time < recording_time:
        success, frame = cam.read()
        ret, jpeg = cv2.imencode(".jpg", frame)
        frame = jpeg.tobytes()
        upload_s3(frame)
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(interval)


@app.route("/video_feed/<int:fps>/<int:recording_time>")
def video_feed(fps, recording_time):
    return Response(
        gen(fps, recording_time), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
