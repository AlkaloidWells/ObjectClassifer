import cv2
import numpy as np
import utils
import constants
import logging


logging.basicConfig(level=logging.INFO)

def orb(img):
    print("in orb")
    # Ensure correct initialization of ORB
    orb = cv2.ORB_create()
    if orb is None:
        print("Failed to create ORB detector.")
        return None

    # Check if the image is correctly loaded
    if img is None:
        print("The image is None. Check the image path and loading function.")
        return None

    # Ensure image is in the correct format
    if len(img.shape) == 2:
        print("Image is grayscale")
    else:
        print("Image is not grayscale. Check if it needs conversion.")

    # Perform ORB detection and computation
    keypoints, descriptors = orb.detectAndCompute(img, None)

    if descriptors is None:
        print("No descriptors found.")
    else:
        print(f"Number of descriptors: {len(descriptors)}")

    return descriptors

def sift(img):
    #print("in sift")
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return descriptors

def gen_codebook(dataset, descriptors, k = 64):
    iterations = 10
    epsilon = 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, epsilon)
    compactness, labels, centers = cv2.kmeans(descriptors, k , None, criteria, iterations, cv2.KMEANS_RANDOM_CENTERS)
    return centers

def vlad(descriptors, centers):
    dimensions = len(descriptors[0])
    vlad_vector = np.zeros((len(centers), dimensions), dtype=np.float32)
    for descriptor in descriptors:
        nearest_center, center_idx = utils.find_nn(descriptor, centers)
        for i in range(dimensions):
            vlad_vector[center_idx][i] += (descriptor[i] - nearest_center[i])
    # L2 Normalization
    vlad_vector = cv2.normalize(vlad_vector, None, norm_type=cv2.NORM_L2)
    vlad_vector = vlad_vector.flatten()
    return vlad_vector

def descriptors_from_class(dataset, class_img_paths, class_number, option=constants.ORB_FEAT_OPTION):
    des = None
    step = (constants.STEP_PERCENTAGE * len(class_img_paths)) / 100
    for i, img_path in enumerate(class_img_paths):
        logging.info(f"Processing image {i+1}/{len(class_img_paths)}: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            logging.warning(f"Failed to read image {img_path}")
            continue

        resize_to = 640
        h, w, channels = img.shape
        if h > resize_to or w > resize_to:
            img = utils.resize(img, resize_to, h, w)

        if option == constants.ORB_FEAT_OPTION:
            des_name = "ORB"
            new_des = orb(img)
        else:
            des_name = "SIFT"
            new_des = sift(img)

        if new_des is not None:
            if des is None:
                des = np.array(new_des, dtype=np.float32)
            else:
                des = np.vstack((des, np.array(new_des, dtype=np.float32)))

        if i % step == 0:
            percentage = (100 * i) / len(class_img_paths)
            logging.info(f"Calculated {des_name} descriptors for image {i} of {len(class_img_paths)} ({percentage:.2f}%) of class number {class_number} ...")

    logging.info(f"* Finished getting the descriptors for the class number {class_number} *")
    logging.info(f"Number of descriptors in class: {len(des)}" if des is not None else "No descriptors found for class")
    dataset.set_class_count(class_number, len(des) if des is not None else 0)
    return des

def all_descriptors(dataset, class_list, option=constants.ORB_FEAT_OPTION):
    des = None
    for i, class_img_paths in enumerate(class_list):
        logging.info(f"*** Getting descriptors for class number {i} of {len(class_list)} ***")
        new_des = descriptors_from_class(dataset, class_img_paths, i, option)
        if des is None:
            des = new_des
        else:
            des = np.vstack((des, new_des))

    logging.info("*****************************")
    logging.info("Finished getting all the descriptors")
    logging.info(f"Total number of descriptors: {len(des)}" if des is not None else "No descriptors found")
    if des is not None and len(des) > 0:
        logging.info(f"Dimension of descriptors: {len(des[0])}")
        logging.info(f"First descriptor:\n{des[0]}")

    return des

if __name__ == "__main__":
    img_path = "Dataset/car/train/vid_4_23340.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {img_path}")
    else:
        print(f"Successfully loaded image: {img_path}")

    descriptors = orb(img)
    print(descriptors)
