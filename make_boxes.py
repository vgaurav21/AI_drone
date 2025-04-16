from picamera2 import Picamera2, Preview
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2

with open('tflite_models/labelmap.txt', 'r') as f:
    labels = {i: line.strip() for i, line in enumerate(f.readlines())}

interpreter = tflite.Interpreter(model_path='tflite_models/detect.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(frame, (300, 300))
    preprocessed_image = resized_frame.astype(np.uint8)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    return preprocessed_image

input_image_path = 'input_image.jpg'
frame = cv2.imread(input_image_path)

preprocessed_image = preprocess_image(frame)

interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
interpreter.invoke()
boxes = interpreter.get_tensor(output_details[0]['index'])[0]
classes = interpreter.get_tensor(output_details[1]['index'])[0]
scores = interpreter.get_tensor(output_details[2]['index'])[0]

threshold = 0.5

for i in range(len(scores)):
    if scores[i] > threshold:
        box = boxes[i]
        class_id = int(classes[i])
        score = scores[i]
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                      ymin * frame.shape[0], ymax * frame.shape[0])
        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
        label = f"{labels[class_id]}: {score:.2f}"
        cv2.putText(frame, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

output_image_path = 'output_image.jpg'
cv2.imwrite(output_image_path, frame)

object_count = sum(1 for score in scores if score > threshold)
print(f'Object Count: {object_count}')
