import tflite_runtime.interpreter as tflite
import os
import cv2
import cvlib as cv
import numpy as np


def read_n_preprocess(image):
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return image


id = input("id를 입력하시오: ")
name = input("이름을 입력하시오: ")

folder_path = f"dataset/{name}"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

count = 1

interpreter = tflite.Interpreter(model_path="C:/jaemin/project/model/my_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)

while True:
    if count > 100:
        break

    ret, frame = cap.read()
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
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 1)

            if count <= 100:
                face_img = frame[start_y:end_y, start_x:end_x]

                if face_img is not None:
                    face_img = read_n_preprocess(face_img)
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(
                        f"{folder_path}/User_"
                        + str(id)
                        + "_"
                        + str(name)
                        + "_"
                        + str(count)
                        + ".jpg",
                        face_img * 255.0,
                    )
                    count += 1

    else:
        label = "Come to Camera"
        color = (0, 0, 255)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 2)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
