from flask import Flask, render_template, Response
import cv2
import threading
import numpy as np
import tensorflow as tf

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


def read_n_preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def train_model():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 640)
    cam.set(4, 480)

    model = tf.keras.models.load_model(
        "C:/jaemin/project/model/face_detection_model1.h5"
    )

    while True:
        ret, frame = cam.read()

        preprocessed_frame = read_n_preprocess(frame)

        predict = model.predict(np.expand_dims(preprocessed_frame, axis=0))[0][0]

        if predict > 0.5:
            label = "Standing There"
            color = (0, 255, 0)
        else:
            label = "COME HERE"
            color = (0, 0, 255)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 2)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cam.release()


@app.route("/train", methods=["POST", "GET"])
def train():
    return Response(
        train_model(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/search")
def search():
    return render_template("search.html")


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
