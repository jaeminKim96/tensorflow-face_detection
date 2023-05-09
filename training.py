import cv2
import os
import numpy as np
from PIL import Image

path = "C:/jaemin/project/dataset"
traindata_path = "C:/jaemin/project/traindata"
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(
    "C:/jaemin/project/haarcascade_frontalface_default.xml"
)

# 이미 얼굴이 꽉 차게 찍게 해서 의미는 없음.


def getImageLabel(path):
    # imagePath = [
    #     os.path.join(path, file)
    #     for file in os.listdir(path)
    #     if os.path.isfile(os.path.join(path, file))
    # ]
    imagePath = []
    for dirpath, dirnames, filenames in os.walk(path):
        for file in filenames:
            if file.endswith(".jpg"):
                imagePath.append(os.path.join(dirpath, file))

    samples = []  # uint8형태로 저장한 것을 dictionary형태로 저장한다.
    labels = []  # id값을 배열로 저장한다.

    for path in imagePath:
        PIL_img = Image.open(path).convert("L")  # convert to grayscale
        numpy_img = np.array(PIL_img, "uint8")
        id = int(os.path.split(path)[-1].split("_")[1])
        faces = detector.detectMultiScale(numpy_img)

        for x, y, w, h in faces:
            samples.append(numpy_img[y : y + h, x : x + w])
            labels.append(id)
    return samples, labels


faces, labels = getImageLabel(path)
recognizer.train(faces, np.array(labels))

if not os.path.exists(traindata_path):
    os.makedirs(traindata_path)

recognizer.save(os.path.join(traindata_path, "traindata.yml"))
print("정보 등록이 완료되었습니다.")
