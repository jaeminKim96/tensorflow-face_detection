import tensorflow as tf
import tensorflow.lite as lite

model = tf.keras.models.load_model("C:/jaemin/project/model/face_detection_model2.h5")

converter = lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("C:/jaemin/project/model/my_model2.tflite", "wb") as f:
    f.write(tflite_model)
