import cv2
import numpy as np
from os import makedirs
from os.path import isdir
import tensorflow as tf
import cvlib as cv
from PIL import Image
from flask import Flask, render_template, Response
import time
import datetime
import json
import requests

app = Flask(__name__)

global quit_flag
quit_flag = False


@app.route("/")
def index():
    return render_template("index.html")


# 전처리용
def read_n_preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # 픽셀 0~1 사이로 정규화
    return img


# 이미지 파일들 저장되는 경로
datasetpath = "C:/jaemin/project/faces/"

# 이미지 파일에서 다시 얼굴 추출(확인용)
face_detector = cv2.CascadeClassifier(
    "C:/jaemin/project/haarcascade_frontalface_default.xml"
)


def take_picture():
    # 개인 별 폴더 생성될 때 사용할 이름
    name = input("이름 입력: ")

    # 폴더 없으면 생성
    if not isdir(datasetpath + name):
        makedirs(datasetpath + name)

    # tensorflow model load
    model = tf.keras.models.load_model(
        "C:/jaemin/project/model/face_detection_final.h5"
    )

    cap = cv2.VideoCapture(0)

    count = 1

    while True:
        if count > 100:
            break

        ret, frame = cap.read()
        if not ret:
            break

        preprocessed_frame = read_n_preprocess(frame)

        prediction = model.predict(np.expand_dims(preprocessed_frame, axis=0))[0][0]

        if prediction > 0.5:  # 사람일 때
            label = "stand there and wait a second"
            color = (0, 255, 0)

            try:
                faces, confidences = cv.detect_face(frame)

                for face in faces:
                    start_x, start_y, end_x, end_y = face

                    cv2.rectangle(
                        frame, (start_x, start_y), (end_x, end_y), (255, 255, 255, 0), 1
                    )

                    if count <= 100:
                        face_img = frame[
                            start_y:end_y, start_x:end_x
                        ]  # detect_face에 의해서 얼굴 영역이 설정됨

                        if face_img is not None:
                            face_img = read_n_preprocess(face_img)
                            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

                            filename_path = (
                                datasetpath + name + "/user" + str(count) + ".jpg"
                            )

                            cv2.imwrite
                            (filename_path, (face_img * 255.0).astype(np.uint8))
                            # print("success")
                            count += 1
                    # else:
                    #    print("save fail")

            except cv2.error as e:
                continue

        else:
            label = "Come to Camera"
            color = (0, 0, 255)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 2)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    print("얼굴 정보 저장이 완료되었습니다")
    cap.release()
    cv2.destroyAllWindows()


@app.route("/detectntrain", methods=["POST", "GET"])
def train():
    return Response(
        take_picture(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/save")
def save():
    return render_template("save.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4412, debug=True, threaded=True)
