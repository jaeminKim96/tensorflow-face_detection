import cv2
import numpy as np
from os import makedirs
from os.path import isdir

# datasetpath
datasetpath = "faces/"

face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# 얼굴 검출
def find_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # 얼굴 안 보이면
    if faces is ():
        return None

    # 얼굴 보이면 그 부분만 좌표 따오기
    for x, y, w, h in faces:
        face_img = img[y : y + h, x : x + w]

    return face_img


# 얼굴 부분만 이미지 파일로 저장하기 -> 이름 전달 받자
def take_picture():
    name = input("입력: ")
    # 폴더가 없으면
    if not isdir(datasetpath + name):
        # 이름으로 폴더 생성하기
        makedirs(datasetpath + name)

    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()

        # 얼굴 검출되면
        if find_face(frame) is not None:
            count += 1
            # 200,200크기로
            face = cv2.resize(find_face(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = datasetpath + name + "/user" + str(count) + ".jpg"
            cv2.imwrite(file_name_path, face)

            cv2.putText(
                face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2
            )
            cv2.imshow("Face Cropper", face)

        else:
            print("face not detected")
            pass

        if cv2.waitKey(1) == ord("q") or count == 100:
            break
    cap.release()
    print("save complete")


if __name__ == "__main__":
    take_picture()
