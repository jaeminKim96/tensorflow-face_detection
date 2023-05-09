import tensorflow as tf
import os
import cv2
import numpy as np


def read_n_preprocess(image):
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return image


# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="C:/jaemin/project/model/my_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open the webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Read each frame of the webcam feed
    ret, frame = cap.read()

    # Preprocess the frame
    preprocessed_frame = read_n_preprocess(frame)

    # Feed the preprocessed frame to the loaded model
    interpreter.set_tensor(
        input_details[0]["index"], np.expand_dims(preprocessed_frame, axis=0)
    )
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])[0][0]

    # Check the predicted output of the model
    if prediction > 0.5:
        label = "Human"
        color = (0, 255, 0)  # Green color for the rectangle

    else:
        label = "Not human"
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
