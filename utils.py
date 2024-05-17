import numpy.random as nprnd
import numpy as np
import cv2
import scipy.io as sio
import matplotlib.pyplot as plt


def random_split(l, sample_size):
    sample_indices = nprnd.choice(len(l), size=sample_size, replace=False)
    # print (len(sample_indices))
    sample_indices.sort()
    # print("sample_indices = {0}".format(sample_indices))
    other_part = []
    sample_part = []
    indices_counter = 0
    for index in range(len(l)):
        current_elem = l[index]
        if indices_counter == sample_size:
            other_part = other_part + l[index:]
            break
        if index == sample_indices[indices_counter]:
            sample_part.append(current_elem)
            indices_counter += 1
        else:
            other_part.append(current_elem)
    return other_part, sample_part

def humanize_time(secs):
    """
    Extracted from http://testingreflections.com/node/6534
    """
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    return '%02d:%02d:%02f' % (hours, mins, secs)

def resize(img, new_size, h, w):
    if h > w:
        new_h = new_size
        new_w = int((new_size * w) / h)
    else:
        new_h = int((new_size * h) / w)
        new_w = new_size
    img = cv2.resize(img, (new_w, new_h))
    return img

def find_nn(point, neighborhood):

    min_dist = float('inf')
    nn = neighborhood[0]
    nn_idx = 0
    for i in range(len(neighborhood)):
        neighbor = neighborhood[i]
        dist = cv2.norm(point - neighbor)
        if dist < min_dist:
            min_dist = dist
            nn = neighbor
            nn_idx = i

    return nn, nn_idx

def save(filename, arr):

    data = {"stored": arr}
    sio.savemat(filename, data)

def load(filename):
    data = sio.loadmat(filename)
    return data["stored"]

def save_csv(filename, arr):

    file = open(filename, "w")
    for row in arr:
        for i in range(len(row) - 1):
            file.write("{0} ".format(row[i]))
        file.write("{0}\n".format(row[len(row) - 1]))

def show_conf_mat(confusion_matrix):
    plt.matshow(confusion_matrix)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.show()