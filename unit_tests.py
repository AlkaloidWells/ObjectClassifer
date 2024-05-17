import cv2
import time
import pickle
import numpy as np
from matplotlib import pyplot as plt

#----------------------------------
# Local dependencies
import descriptors
import utils
from dataset import Dataset
import constants
import filenames


svm_kernel = cv2.ml.SVM_LINEAR


import cv2
import numpy as np
import utils
import descriptors
import filenames


def test_one_img_classification(image_path):
    img = cv2.imread(image_path)
    resize_to = 640
    h, w, channels = img.shape
    img = utils.resize(img, resize_to, h, w)
    des = descriptors.sift(img)
    k = 128
    des_name = "SIFT"
    codebook_filename = filenames.codebook(k, des_name)
    codebook = utils.load(codebook_filename)
    img_vlad = descriptors.vlad(des, codebook)

    # Ensure img_vlad is reshaped correctly and is of type float32
    if img_vlad.ndim == 1:
        img_vlad = img_vlad.reshape(1, -1)  # Reshape to (1, 8192)
    
    img_vlad = np.asarray(img_vlad, dtype=np.float32)

    # Load and use the SVM
    svm_filename = filenames.svm(k, des_name, svm_kernel)
    svm = cv2.ml.SVM_load(svm_filename)

    # Predict using the SVM
    _, result = svm.predict(img_vlad)
    result = result.flatten()  # Flatten if necessary

    # Map numerical results to class names
    class_names = {
        0: "Dog",
        1: "Car",
        2: "Human",
        3: "Cat",
        4: "Horse"
    }

    # Get the class name(s) corresponding to the result
    predicted_classes = [class_names[int(res)] for res in result]
    #print("Predicted class(es):", predicted_classes)

    return predicted_classes

def test_one_image_pretrained(path_image):
    model = "Pretrained/bvlc_googlenet.caffemodel"
    protxt = "Pretrained/bvlc_googlenet.prototxt.txt"


    net = cv2.dnn.readNetFromCaffe(protxt, model)

    text_file = open("Pretrained/classification_classes_ILSVRC2012.txt", "r")
    lines = text_file.readlines()

    # Correct the file path for Unix-based systems (Ubuntu)
    frame = cv2.imread(path_image)

    if frame is None:
        raise FileNotFoundError("Image not found. Check the file path and ensure the image exists.")

    frame = cv2.resize(frame, (400, 300))

    model_frame = cv2.resize(frame, (224, 224))
    blobfromImage = cv2.dnn.blobFromImage(model_frame, 1, (224, 224))
    net.setInput(blobfromImage)
    classifications = net.forward()

    min_value, max_value, min_loc, max_loc = cv2.minMaxLoc(classifications)

    class_probability = max_value
    class_number = max_loc

    if class_probability > 0.2:
        label = lines[class_number[0]].strip()
        
    else:
        label = "unknown"
       
    return label
    



if __name__ == '__main__':
    image_path1 = "test/test3.jpg"

    result1 = test_one_img_classification(image_path1)
    result2 = test_one_image_pretrained(image_path1)
    
    # Join the list of predicted classes into a single string
    result1_str = ", ".join(result1)
    print("Prediction by Trained Model: " + result1_str)

    result1_str = "".join(result2)
    print("Prediction by Pretrained Model: " + result1_str)