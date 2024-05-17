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
from main1 import yolo


svm_kernel = cv2.ml.SVM_LINEAR


import cv2
import numpy as np
import utils
import descriptors
import filenames
from collections import Counter
import re
import nltk
from nltk import pos_tag

# Download the NLTK data for POS tagging (if not already downloaded)
#nltk.download('averaged_perceptron_tagger')

def find_common_noun(input_string):
    # Check if the input string is a single word
    if ' ' not in input_string:
        tagged_word = pos_tag([input_string])
        if any(tag.startswith('NN') for word, tag in tagged_word):
            return input_string
        else:
            return "No common noun found"

    # Split the input string by commas and whitespace into a list of words
    words = input_string.split()
    
    # Perform POS tagging on the words
    tagged_words = pos_tag(words)
    
    # Filter out words that are not nouns
    nouns = [word for word, pos in tagged_words if pos.startswith('NN')]
    
    # If no nouns are found, return "No common noun found"
    if not nouns:
        return "No common noun found"
    
    # Return the most common noun
    return max(nouns, key=nouns.count)




def split(input_string):
    # Split the string by commas
    words = input_string.split(',')
    # Strip any leading/trailing whitespace and return the first word
    first_word = words[0].strip()
    return first_word



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

    #print(classifications)
    min_value, max_value, min_loc, max_loc = cv2.minMaxLoc(classifications)

    class_probability = max_value
    class_number = max_loc

    if class_probability > 0.2:
        label = lines[class_number[0]].strip()
        print(class_probability)
        
    else:
        label = "unknown"
       
    return label
    



if __name__ == '__main__':
    image_path1 = "test/sh22.jpg"
    print(image_path1)
    result1 = test_one_img_classification(image_path1)
    result2 = test_one_image_pretrained(image_path1)
    result3 = yolo(image_path1)


    result1_str1 = "".join(result1)
    result1_str2 = "".join(result2)
    result1_str3 = str(result3)
    # Join the list of predicted classes into a single string
   
    if result1_str2 == "unknown":
        if result1_str3 == "None":
            print("unknown_item")
        else:
            result1_str2 = result1_str3
    else:
        if result1_str3 == "None":
            #result1_str3 = split(result1_str2)       
            result1_str3 = find_common_noun(result1_str2)
        else:
            result1_str2 = (result1_str3 +  ", " + result1_str2)
            result1_str3 =  find_common_noun(result1_str3)


    #print("Prediction by Trained Model: " + result1_str1)
    print("Product Catigoury: " + result1_str3)
    print("Product Name: " + result1_str2)
    

