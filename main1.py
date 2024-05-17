import cv2
import numpy as np


def get_network(weights, configuration, image):
    network = cv2.dnn.readNet(weights, configuration)
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416))
    network.setInput(blob)
    return network


def get_classes(path):
    with open(path, 'r') as file:
        return [line.strip() for line in file.readlines()]


def get_output_layers(network):
    names = network.getLayerNames()
    return [names[i - 1] for i in network.getUnconnectedOutLayers()]


def predict(network, height, width):
    prediction_layers = network.forward(get_output_layers(network))
    confidences = []
    boxes = []
    class_indices = []
    for prediction_layer in prediction_layers:
        for prediction in prediction_layer:
            scores = prediction[5:]
            class_index = np.argmax(scores)
            class_indices.append(class_index)
            confidences.append(scores[class_index])
            center_x, center_y, w, h = prediction[:4]
            boxes.append([center_x - w / 2, center_y - h / 2, w, h])
    suppressed_indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    suppressed_boxes = []
    suppressed_class_indices = []
    for i in suppressed_indices:
        suppressed_boxes.append([int(boxes[i][0] * width), int(boxes[i][1] * height),
                                 int(boxes[i][2] * width), int(boxes[i][3] * height)])
        suppressed_class_indices.append(class_indices[i])
    return suppressed_boxes, suppressed_class_indices


def draw_object(image, box, class_name, height):
    x, y, w, h = box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    text_width, text_height = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
    text_x = int(x + (w - text_width) / 2)
    text_y = y - 10
    if text_x < 0:
        text_x = x
    if text_y < 0:
        text_y = y + h + text_height
        if text_y >= height:
            text_y = y
    cv2.putText(image, class_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

def yolo(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    network = get_network('Pretrained/yolo.weights', 'Pretrained/yolo.cfg', image)
    classes = get_classes('Pretrained/yolo.names')
    boxes, class_indices = predict(network, height, width)
    count = {}
    for box, class_index in zip(boxes, class_indices):
        draw_object(image, box, classes[class_index], height)
        if class_index in count:
            count[class_index] += 1
        else:
            count[class_index] = 1
    if count:
        print('Detected:')
    for class_index in count:
        return str(classes[class_index])
    
