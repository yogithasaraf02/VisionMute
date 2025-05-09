import cv2
import numpy as np
import os
from os.path import join, isfile
from matplotlib import pyplot as plt
from google.colab import files
import zipfile


class ShushDetector:
    def __init__(self, face_cascade_path, mouth_cascade_path):
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)

        if self.face_cascade.empty() or self.mouth_cascade.empty():
            raise IOError("Could not load one or both cascade classifiers.")

    def remove_nested_boxes(self, boxes):
        final_boxes = []
        inside_boxes = []

        for i, box1 in enumerate(boxes):
            for j, box2 in enumerate(boxes):
                if i == j:
                    continue
                x1, y1, x1b, y1b = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
                x2, y2, x2b, y2b = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
                if (x1 < x2 + 3 and y1 < y2 + 3 and x1b > x2b - 3 and y1b > y2b - 3):
                    inside_boxes.append(box2.tolist())

        for box in boxes:
            if box.tolist() not in inside_boxes:
                final_boxes.append(box)
        return final_boxes

    def detect_mouth(self, frame, roi_origin, roi):
        mouths = self.mouth_cascade.detectMultiScale(
            roi,
            scaleFactor=1.5,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(10, 10)
        )
        mouths = self.remove_nested_boxes(mouths)
        for (mx, my, mw, mh) in mouths:
            abs_x, abs_y = mx + roi_origin[0], my + roi_origin[1]
            cv2.rectangle(frame, (abs_x, abs_y), (abs_x + mw, abs_y + mh), (0, 0, 255), 2)
        return len(mouths) == 0 

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=4,
            minSize=(40, 40)
        )
        detected = 0

        if len(faces) == 0:
            print("No face found")
            if self.detect_mouth(image, (0, 0), gray):
                print("Shush detected (no face)")
                detected += 1
            else:
                print("Shush not detected (no face)")
        else:
            faces = self.remove_nested_boxes(faces)
            for (x, y, w, h) in faces:
                y_mouth = y + h // 2
                mouth_roi = gray[y_mouth:y + h, x:x + w]
                if self.detect_mouth(image, (x, y_mouth), mouth_roi):
                    print("Shush detected")
                    detected += 1
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                else:
                    print("Shush not detected")
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image, detected


def show_image(img, title='Detected'):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


def process_zip_and_detect(zip_filename, detector):
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall("extracted_images")
    
    files = [f for f in os.listdir("extracted_images") if isfile(join("extracted_images", f))]
    total_detections = 0

    for file in files:
        path = join("extracted_images", file)
        img = cv2.imread(path)
        if img is not None:
            result_img, detections = detector.detect(img)
            total_detections += detections
            show_image(result_img, title=file)
    print("Total detections:", total_detections)


print("Upload a ZIP file with images:")
uploaded = files.upload()

zip_file = next(iter(uploaded))
face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
mouth_path = "mouth.xml" 

if not os.path.exists(mouth_path):
    print("Upload your Mouth.xml:")
    uploaded = files.upload()
    mouth_path = next(iter(uploaded))

detector = ShushDetector(face_path, mouth_path)
process_zip_and_detect(zip_file, detector)
