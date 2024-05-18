import cv2
import numpy as np
from matplotlib import pyplot as plt
import descriptors
import utils
import filenames
from main1 import yolo
import cv2
import numpy as np
import inflect
import utils
import descriptors
import filenames
from nltk import pos_tag
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn
import os
import glob

svm_kernel = cv2.ml.SVM_LINEAR
p = inflect.engine()




# Download the NLTK data for POS tagging (if not already downloaded)
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')




def get_file_paths(folder_path):
    # List to store file paths
    file_paths = []

    # Use glob to find all files in the folder
    for file in glob.glob(os.path.join(folder_path, '*')):
        # Append the full path to the file_paths list
        file_paths.append(os.path.abspath(file))

    return file_paths

# Example usage:
folder_path = '/path/to/your/folder'  # Replace with your folder path
file_paths = get_file_paths(folder_path)
for path in file_paths:
    print(path)


def remove_duplicates(words):
    # Split the string into words based on commas
    word_list = words.split(',')
    
    # Remove leading and trailing whitespaces from each word
    word_list = [word.strip() for word in word_list]
    
    # Use a set to store unique words while preserving the order
    unique_words = []
    seen = set()
    for word in word_list:
        if word not in seen:
            unique_words.append(word)
            seen.add(word)
    
    # Join the unique words back into a string separated by commas
    result = ', '.join(unique_words)
    
    return result

def is_countable_noun(word):
    # Check if the word is a noun according to WordNet
    synsets = wn.synsets(word, pos=wn.NOUN)
    if synsets:
    
        # Check if the word is already in its plural form
        if p.singular_noun(word):
            plural_form = word
        else:
            plural_form = p.plural(word)

        # Check if the plural form is in the cat.txt file
        with open('Pretrained/cat.txt', 'r') as file:
            cat_words = [line.strip().lower() for line in file]
        if plural_form.lower() in cat_words:
            return True
    return False


def split_and_check_noun(word):
    """Split a word and check for countable nouns in its components."""
    fragments = [word[i:j] for i in range(len(word)) for j in range(i + 1, len(word) + 1) if len(word[i:j]) >= 3]
    countable_nouns = [fragment for fragment in fragments if is_countable_noun(fragment)]
    # Remove the original word from the list of countable nouns
    countable_nouns = [noun for noun in countable_nouns if noun != word]
    return countable_nouns


def find_common_noun(input_string):
    noun_counts = {}
    # Check if the input string is a single word
    if ' ' not in input_string and ',' not in input_string:
        if is_countable_noun(input_string):
            noun_counts[input_string] = noun_counts.get(input_string, 0) + 1
            if len(input_string) >=3:
                countable_nouns = split_and_check_noun(input_string)
                for word2 in countable_nouns:
                    
                    if is_countable_noun(word2):
                        noun_counts[word2] = noun_counts.get(word2, 0) + 1

                                         
        return max(noun_counts, key=noun_counts.get)

    # Tokenize and split the input string by commas and whitespace into a list of words
    words = [word.strip() for word in input_string.replace(',', ' ').split()]
    # Perform POS tagging on the words
    tagged_words = pos_tag(words)
    
    # Initialize a dictionary to keep track of countable nouns and their counts
    

    # Check each word and its components
    for word, pos in tagged_words:
        if is_countable_noun(word):
            noun_counts[word] = noun_counts.get(word, 0) + 1
            countable_nouns = split_and_check_noun(word)
            
            for word1 in countable_nouns:
                
                if is_countable_noun(word1):
                    noun_counts[word1] = noun_counts.get(word1, 0) + 1

        else:
            countable_nouns = split_and_check_noun(word)
        
            for word1 in countable_nouns:
                if is_countable_noun(word1):
                    noun_counts[word1] = noun_counts.get(word1, 0) + 1

    

    
    # If no countable nouns are found, return "No common category found"
    if not noun_counts:
        return "No common category found"

    # Return the countable noun with the highest count
    return max(noun_counts, key=noun_counts.get)



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
        class_probability = 0
       
    return class_probability, label
    

def Classifer(image_path):
    print(image_path)
    result1 = test_one_img_classification(image_path)
    c1, result2 = test_one_image_pretrained(image_path)
    result3 = yolo(image_path)
    #print(result1)
    print(result2)
    print(result3)

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
            if c1 > 0.5:
                result1_str2 = (result1_str3 +  ", " + result1_str2)
                result1_str3 =  find_common_noun(result1_str3 +  ", " + result1_str2)
                result1_str2 = remove_duplicates(result1_str3 +  ", " + result1_str2)
            else:
                result1_str2 = (result1_str3 +  ", " + result1_str2)
                result1_str3 =  find_common_noun(result1_str3 +  ", " + result1_str2)

    return result1_str2, result1_str3

if __name__ == '__main__':

    paths = get_file_paths("test/")
    for path in paths:
        rec2, rec3 = Classifer(path)    
        #print("Prediction by Trained Model: " + result1_str1)
        print("Product Catigoury: " + rec3)
        print("Product Name: " + rec2)
        print("\n")
        print("*****************************************")
        print("\n")


 


