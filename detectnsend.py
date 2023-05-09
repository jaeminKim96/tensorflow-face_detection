import tensorflow as tf
import os
import cv2
import numpy as np
import cvlib as cv
import mysql.connector


def read_n_preprocess(image):
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return image


id = input("id를 입력하세요: ")
name = input("이름을 입력하세요: ")
folder_path = f"dataset/{name}"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# mysql db에 전송
sender = mysql.connector.connect(
    host="localhost", password="11010331", database="test", user="root"
)

cursor = sender.cursor()
count = 1

# Load the saved model
model = tf.keras.models.load_model("C:/jaemin/project/model/face_detection_model1.h5")

# Open the webcam feed
cap = cv2.VideoCapture(0)

while True:
    if count > 5:
        break
    # Read each frame of the webcam feed
    ret, frame = cap.read()

    # Preprocess the frame
    preprocessed_frame = read_n_preprocess(frame)

    # Feed the preprocessed frame to the loaded model
    prediction = model.predict(np.expand_dims(preprocessed_frame, axis=0))[0][0]

    # Check the predicted output of the model
    if prediction > 0.5:
        label = "Stand there and wait"
        color = (0, 255, 0)  # Green color for the rectangle
        faces, confdences = cv.detect_face(frame)

        for face in faces:
            start_x, start_y, end_x, end_y = face
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

            if count <= 100:
                face_img = frame[start_y:end_y, start_x:end_x]
                if face_img is not None:
                    # img_copy = np.copy(frame)

                    # cv2.rectangle(
                    #     img_copy, (start_x, start_y), (end_x, end_y), color, 2
                    # )

                    face_img = read_n_preprocess(face_img)
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

                    # 파일 이름 정리
                    # cv2.imwrite(f"{folder_path}/{name}.{count}.jpg", face_img * 255.0)
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
                    # cv2.imwrite(
                    #     folder_path.encode("utf-8").decode("utf-8")
                    #     + "/User_"
                    #     + str(id)
                    #     + "_"
                    #     + str(name)
                    #     + "_"
                    #     + str(count)
                    #     + ".jpg",
                    #     face_img * 255.0,
                    # )

                    # if count == 1:
                    #     query = f"INSERT INTO practice (id, name, pic) VALUES ({id},'{name}','dataset/{name}/User_{id}_{name}_1.jpg')"

                    #     cursor.execute(query)

                    #     sender.commit()

                    count += 1

    else:
        label = "Come to camera"
        color = (0, 0, 255)  # Red color for the rectangle

    # Draw a rectangle around the detected face (if any)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 2)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the frame on the screen
    cv2.imshow("Face Detection", frame)

    # Wait for a key press to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam feed and destroy all windows
cap.release()
cv2.destroyAllWindows()
