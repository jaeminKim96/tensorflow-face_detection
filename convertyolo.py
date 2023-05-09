import torch
import tensorflow as tf
from torch.autograd import Variable

model = torch.load("C:/Users/user/Desktop/yolov5-master/jmbest.pt")

example = Variable(torch.rand(1, 3, 416, 416))
traced_model = torch.jit.trace(model, example)

input_signature = tf.TensorSpec([1, 3, 416, 416], dtype=tf.float32)
converter = tf.lite.TFLiteConverter.from_pytorch(
    traced_model, input_signature=input_signature
)
tflite_model = converter.convert()

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

with open("quantized_yolo.tflite", "wb") as f:
    f.write(tflite_quantized_model)
