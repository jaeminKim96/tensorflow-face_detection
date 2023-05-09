import cv2
import numpy as np
from os import listdir
from os.path import isdir, isfile, join

# 얼굴 인식용 haar/cascade 로딩
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# 얼굴 학습
def train(name):
    data_path = "faces/" + name + "/"

    # file만 list로 생성
    pic_list = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    train_data, labels = [], []

    for i, files in enumerate(pic_list):
        img_path = data_path + pic_list[i]
        imgs = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if imgs is None:
            continue

        train_data.append(np.asarray(imgs, dtype=np.uint8))
        labels.append(i)

    if len(labels) == 0:
        print("no data to train")
        return None

    labels = np.asarray(labels, dtype=np.int32)
    model = cv2.face.LBPHFaceRecognizer_create()

    model.train(np.asarray(train_data), np.asarray(labels))

    print(name + "training is completed")

    return model


# 여러 얼굴 학습
def trains():
    data_path = "faces/"

    model_dir = [f for f in listdir(data_path) if isdir(join(data_path, f))]

    # 결과 저장할 딕셔너리
    results = {}

    for model in model_dir:
        print("model: " + model)

        # 학습
        result = train(model)

        # 학습 안 되면
        if result is None:
            continue

        # 학습되었다면 저장

        results[model] = result

    return results


# 얼굴 검출
def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return img, []

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        roi = img[y : y + h, x : x + w]
        roi = cv2.resize(roi, (200, 200))

    return img, roi


def run(models):
    # 카메라 열기
    cap = cv2.VideoCapture(0)

    while True:
        # 카메라로 부터 사진 한장 읽기
        ret, frame = cap.read()
        # 얼굴 검출 시도
        image, face = face_detector(frame)
        try:
            min_score = 999  # 가장 낮은 점수로 예측된 사람의 점수
            min_score_name = ""  # 가장 높은 점수로 예측된 사람의 이름

            # 검출된 사진을 흑백으로 변환
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # 위에서 학습한 모델로 예측시도
            for key, model in models.items():
                result = model.predict(face)
                if min_score > result[1]:
                    min_score = result[1]
                    min_score_name = key

            # min_score 신뢰도이고 0에 가까울수록 자신과 같다는 뜻이다.
            if min_score < 500:
                # ????? 어쨋든 0~100표시하려고 한듯
                confidence = int(100 * (1 - (min_score) / 300))
                # 유사도 화면에 표시
                display_string = (
                    str(confidence) + "% Confidence it is " + min_score_name
                )
            cv2.putText(
                image,
                display_string,
                (100, 120),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (250, 120, 255),
                2,
            )
            # 75 보다 크면 동일 인물로 간주해 UnLocked!
            if confidence > 56:
                cv2.putText(
                    image,
                    "Unlocked : " + min_score_name,
                    (250, 450),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Face Cropper", image)
            else:
                # 75 이하면 타인.. Locked!!!
                cv2.putText(
                    image,
                    "Locked",
                    (250, 450),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Face Cropper", image)
        except:
            # 얼굴 검출 안됨
            cv2.putText(
                image,
                "Face Not Found",
                (250, 450),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 0, 0),
                2,
            )
            cv2.imshow("Face Cropper", image)
            pass
        if cv2.waitKey(1) == 13:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 학습 시작
    models = trains()
    # 고!
    run(models)
