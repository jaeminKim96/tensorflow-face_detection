import tflite_runtime.interpreter as tflite
import os
import cv2
import cvlib as cv
import numpy as np
from PIL import Image
from flask import Flask, render_template, Response
import datetime
import time
import json
import requests

app = Flask(__name__)

global quit_flag
quit_flag = False


@app.route("/")
def index():
    return render_template("index.html")


def read_n_preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def for_detect_train():
    cam = cv2.VideoCapture(0)

    datasetpath = "dataset1"
    traindatapath = "traindata1"

    face_id = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    face_detector = cv2.CascadeClassifier(
        "C:/jaemin/project/haarcascade_frontalface_default.xml"
    )

    interpreter = tflite.Interpreter(
        model_path="C:/jaemin/project/model/my_model.tflite"
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    count = 1

    while True:
        if count > 100:
            break

        ret, frame = cam.read()
        if not ret:
            break

        preprocessed_frame = read_n_preprocess(frame)

        interpreter.set_tensor(
            input_details[0]["index"], np.expand_dims(preprocessed_frame, axis=0)
        )
        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]["index"])

        if prediction > 0.5:
            label = "stand there and wait a second"
            color = (0, 255, 0)

            faces, confidences = cv.detect_face(frame)

            for face in faces:
                start_x, start_y, end_x, end_y = face
                cv2.rectangle(
                    frame, (start_x, start_y), (end_x, end_y), (255, 255, 255, 0), 1
                )

                if count <= 100:
                    face_img = frame[start_y:end_y, start_x:end_x]

                    if face_img is not None:
                        face_img = read_n_preprocess(face_img)
                        face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

                        cv2.imwrite(
                            f"{datasetpath}/User_{face_id}.{count}.jpg",
                            face_img * 255.0,
                        )
                        count += 1

        else:
            label = "Come to Camera"
            color = (0, 0, 255)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 2)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    print("얼굴 정보 저장이 완료되었습니다")
    cam.release()
    # cv2.destroyAllWindows()
    print("얼굴 정보를 학습 중입니다\n 잠시만 기다려 주십시오...")

    # training
    def getImageLabel(path):
        img_paths = [os.path.join(path, f) for f in os.listdir(path)]

        face_samples = []
        labels = []

        for img_path in img_paths:
            PIL_img = Image.open(img_path).convert("L")
            img_numpy = np.array(PIL_img, "uint8")
            label = int(os.path.split(img_path)[-1].split(".")[1])
            faces = face_detector.detectMultiScale(img_numpy)

            for x, y, w, h in faces:
                face_samples.append(img_numpy[y : y + h, x : x + w])
                labels.append(label)

        return face_samples, labels

    faces, labels = getImageLabel(datasetpath)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    if not os.path.exists(traindatapath):
        os.makedirs(traindatapath)

    recognizer.save(os.path.join(traindatapath, "traindata.yml"))
    print("얼굴 등록 완료!")


def recognize_frames():
    global quit_flag
    quit_flag = False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("C:/jaemin/project/traindata1/traindata.yml")

    font = cv2.FONT_HERSHEY_SIMPLEX

    cam = cv2.VideoCapture(1)

    cam.set(3, 640)
    cam.set(4, 480)

    while True:
        if quit_flag == True:
            break

        success, frame = cam.read()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces, confidences = cv.detect_face(rgb)

        for (x, y, w, h), conf in zip(faces, confidences):
            cv2.rectangle(frame, (x, y), (w, h), (255, 255, 255, 128), 2)

            try:
                gray = cv2.cvtColor(frame[y:h, x:w], cv2.COLOR_BGR2GRAY)

            except cv2.error:
                continue

            id, confidence = recognizer.predict(gray)

            if confidence < 55:
                id = "Member"
                datas = {
                    "data": {
                        "visitant": "member",
                        "visit_time": datetime.datetime.now().strftime(
                            "%Y-%m-%d-%H-%M"
                        ),
                    }
                }

                url = "http://13.124.30.108:8080/insert/Visitor"
                response = requests.post(url, data=json.dumps(datas))
                print(response.json())
            else:
                id = "Unknown"

            cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (0, 255, 0, 2))

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        # Flask 애플리케이션에서 비디오 프레임을 반환하는 제너레이터
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cam.release()


@app.route("/detectntrain", methods=["POST", "GET"])
def train():
    return Response(
        for_detect_train(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/save")
def save():
    return render_template("save.html")


@app.route("recognize", methods=["POST", "GET"])
def recognizing():
    return Response(
        recognize_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/searching")
def searching():
    return render_template("search.html")


@app.route("/releasing")
def camrelease():
    global quit_flag
    quit_flag = True
    time.sleep(1)
    return "ok"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4412, debug=True, threaded=True)
