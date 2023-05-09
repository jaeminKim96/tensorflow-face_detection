import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(
    "C:/jaemin/project/model/face_detection_model1.h5"
)

converter.target_spec.supported_types = [tf.int8]
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("C:/jaemin/project/model/quantizemodel.tflite", "wb") as f:
    f.write(tflite_model)
