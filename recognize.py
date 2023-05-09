import cv2
import numpy as np
import os
import cvlib as cv
from cvlib.object_detection import draw_bbox

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("C:/jaemin/project/traindata/traindata.yml")

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

names = [
    name
    for name in os.listdir("dataset")
    if os.path.isdir(os.path.join("dataset", name))
]
names.insert(0, "None")

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces, confidences = cv.detect_face(rgb)

    for (x, y, w, h), conf in zip(faces, confidences):
        cv2.rectangle(frame, (x, y), (w, h), (255, 255, 255), 2)

        try:
            gray = cv2.cvtColor(frame[y:h, x:w], cv2.COLOR_BGR2GRAY)
        except cv2.error:
            continue

        id, confidence = recognizer.predict(gray)

        if confidence < 55:
            id = names[id]
            confidence = "{0}%".format(round(100 - confidence))

        else:
            id = "unkown"
            confidence = "{0}%".format(round(100 - confidence))

        cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (0, 255, 0, 2))
        # cv2.putText(frame, str(confidence), (x + 5, y + h - 5), font, 1, (0, 255, 0), 2)

    cv2.imshow("recognize", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

print("종료되었습니다.")
cap.release()
cv2.destroyAllWindows()
