import cv2
import numpy as np
import time


# Local dependencies

import constants
import descriptors
import filenames
import utils


class Classifier:
 
    def __init__(self, dataset, log):

        self.dataset = dataset
        self.log = log

   

    def train(self, svm_kernel, codebook, des_option=constants.ORB_FEAT_OPTION, is_interactive=True):
        des_name = constants.ORB_FEAT_NAME if des_option == constants.ORB_FEAT_OPTION else constants.SIFT_FEAT_NAME
        k = len(codebook)
        x_filename = filenames.vlads_train(k, des_name)
        
        if is_interactive:
            data_option = int(input("Enter [1] to calculate VLAD vectors for the training set or [2] to load them.\n"))
        else:
            data_option = constants.GENERATE_OPTION
        
        if data_option == constants.GENERATE_OPTION:
            print("Getting global descriptors for the training set.")
            start = time.time()
            x, y = self.get_data_and_labels(self.dataset.get_train_set(), codebook, des_option)
            utils.save(x_filename, x)
            end = time.time()
            print("VLADs training vectors saved on file {0}".format(x_filename))
            self.log.train_vlad_time(end - start)
        else:
            print("Loading global descriptors for the training set.")
            x = utils.load(x_filename)
            y = self.dataset.get_train_y()
            x = np.matrix(x, dtype=np.float32)

        print(f"Data loaded. x shape: {x.shape}, y shape: {y.shape}")

        if x.size == 0 or y.size == 0:
            raise ValueError("Loaded data is empty. Please check the data source.")
        
        y = y.reshape(-1, 1).astype(np.int32)  # Ensure y is an integer type
        print(f"Data reshaped. x shape: {x.shape}, y shape: {y.shape}")

        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Data contains NaNs or infinite values. Please check the data source.")

        print(f"First few values of y: {y[:10].flatten()}")
        unique_labels = np.unique(y)
        print(f"Unique labels in y: {unique_labels}")

        if len(unique_labels) < 2:
            raise ValueError("Not enough unique labels in y. The SVM needs at least two different labels to function properly.")

        svm = cv2.ml.SVM_create()
        svm.setKernel(svm_kernel)
        svm.setType(cv2.ml.SVM_C_SVC)

        svm_filename = filenames.svm(k, des_name, svm_kernel)
        
        if is_interactive:
            svm_option = int(input("Enter [1] for generating a SVM or [2] to load one\n"))
        else:
            svm_option = constants.GENERATE_OPTION
        
        if svm_option == constants.GENERATE_OPTION:
            print("Calculating the Support Vector Machine for the training set...")
            svm.setC(1.0)
            svm.setGamma(0.01)

            print(f"x mean: {np.mean(x)}, x std: {np.std(x)}, x min: {np.min(x)}, x max: {np.max(x)}")
            print(f"y mean: {np.mean(y)}, y std: {np.std(y)}, y min: {np.min(y)}, y max: {np.max(y)}")

            print(x)
            
            start = time.time()
            try:
                # Attempt training with simpler parameters
                print("Starting SVM training with simplified parameter grids.")
                svm.train(x, cv2.ml.ROW_SAMPLE, y)
            except cv2.error as e:
                print(f"OpenCV error during train: {e}")
                raise
            end = time.time()
            self.log.svm_time(end - start)

            svm.save(svm_filename)
        else:
            svm = cv2.ml.SVM_load(svm_filename)
        
        return svm




    def test(self, codebook, svm, des_option=constants.ORB_FEAT_OPTION, is_interactive=True):

        des_name = constants.ORB_FEAT_NAME if des_option == constants.ORB_FEAT_OPTION else constants.SIFT_FEAT_NAME
        k = len(codebook)
        x_filename = filenames.vlads_test(k, des_name)
        if is_interactive:
            data_option = int(input("Enter [1] to calculate VLAD vectors for the testing set or [2] to load them.\n"))
        else:
            data_option = constants.GENERATE_OPTION
        if data_option == constants.GENERATE_OPTION:
            # Getting the global vectors for all of the testing set
            print("Getting global descriptors for the testing set...")
            start = time.time()
            x, y = self.get_data_and_labels(self.dataset.get_test_set(), codebook, des_option)
            utils.save(x_filename, x)
            end = time.time()
            print("VLADs testing vectors saved on file {0}".format(x_filename))
            self.log.test_vlad_time(end - start)
        else:
            # Loading the global vectors for all of the testing set
            print("Loading global descriptors for the testing set.")
            x = utils.load(x_filename.format(des_name))
            y = self.dataset.get_test_y()
            x = np.matrix(x, dtype=np.float32)

        # Predicting the testing set using the SVM
        start = time.time()
        _, result = svm.predict(x)  # Use svm.predict
        result = result.flatten()  # Flatten the result to match the shape of y if necessary
        end = time.time()
        self.log.predict_time(end - start)

        mask = result == y
        correct = np.count_nonzero(mask)
        accuracy = (correct * 100.0 / result.size)
        print(accuracy)
        self.log.accuracy(accuracy)
        return result, y

    def get_data_and_labels(self, img_set, codebook, des_option = constants.ORB_FEAT_OPTION):

        y = []
        x = None
        for class_number in range(len(img_set)):
            img_paths = img_set[class_number]
            step = round(constants.STEP_PERCENTAGE * len(img_paths) / 100)
            for i in range(len(img_paths)):
                if (step > 0) and (i % step == 0):
                    percentage = (100 * i) / len(img_paths)
                    print("Calculating global descriptors for image number {0} of {1}({2}%)".format(
                        i, len(img_paths), percentage)
                    )
                img = cv2.imread(img_paths[i])
                if des_option == constants.ORB_FEAT_OPTION:
                    des = descriptors.orb(img)
                else:
                    des = descriptors.sift(img)
                if des is not None:
                    des = np.array(des, dtype=np.float32)
                    vlad_vector = descriptors.vlad(des, codebook)
                    if x is None:
                        x = vlad_vector
                        y.append(class_number)
                    else:
                        x = np.vstack((x, vlad_vector))
                        y.append(class_number)
                else:
                    print("Img with None descriptor: {0}".format(img_paths[i]))
        y = np.float32(y)[:, np.newaxis]
        x = np.matrix(x)
        return x, y