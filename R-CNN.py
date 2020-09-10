
## Original code from pyimagesearch ##
## https://www.pyimagesearch.com/category/deep-learning/ ##
## "Turning any CNN image classifier into an object detector with Keras, Tensorflow, and OpenCV"  ##
## "Intersection over Union (IoU) for object detection" ##

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras import models, layers
import tensorflow.keras.backend as K
from imutils.object_detection import non_max_suppression
from collections import namedtuple
import numpy as np
from sklearn.utils import shuffle
import argparse
import imutils
import time
import cv2
import os
import xmltodict
import json


def get_args():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, help="path to the input images")
    ap.add_argument("-s", "--size", type=str, default="(64, 64)", help="ROI size (in pixels)")
    ap.add_argument("-c", "--min-conf", type=float, default=0.9, help="minimum probability to filter weak detections")
    ap.add_argument("-v", "--visualize", type=int, default=-1, help="whether or not to show extra visualizations for debugging")
    args = vars(ap.parse_args())
    return args

def get_data(path):
    images_path = path + 'JPEGImages/'
    annotations_path = path + 'Annotations/'

    images = []
    for im in os.listdir(images_path):
        images.append(cv2.imread(images_path + im))
        
    annotations = []
    for an in os.listdir(annotations_path):
        with open(annotations_path + an, 'rb') as f:
            annotations.append(xmltodict.parse(f)['annotation']['object'])

    labels = []
    bounding_boxes = []
    for annotation in annotations:
        for cell in annotation:
            labels.append(cell['name'])
            bb = [cell['bndbox'][key] for key in cell['bndbox'].keys()]
            bounding_boxes.append(bb)
    bounding_boxes = np.array(bounding_boxes, dtype='int32')

    return (images, bounding_boxes, labels)


class RCNN:

    def __init__(self, args):
        self.WIDTH = 600
        self.PYR_SCALE = 1.5    # default: 1.5
        self.MIN_SIZE = (2000, 2000)
        self.ROI_SIZE = (224, 224)
        self.WIN_STEP = int(self.ROI_SIZE[0] / 4)
        self.INPUT_SIZE = eval(args["size"])
        self.make_RPN()

    def sliding_window(self, image, step, ws):
        # slide a window across the image
        for y in range(0, image.shape[0] - ws[1], step):
            for x in range(0, image.shape[1] - ws[0], step):
                # yield the current window
                yield (x, y, image[y:y + ws[1], x:x + ws[0]])
                if y + ws[1]*2 < image.shape[0]:
                    yield (x, y, image[y:y + ws[1]*2, x:x + ws[0]])
                if x + ws[0]*2 < image.shape[1]:
                    yield (x, y, image[y:y + ws[1], x:x + ws[0]*2])

    def image_pyramid(self, image, scale, minSize):
        # yield the original image
        yield image
        # keep looping over the image pyramid
        while True:
            # compute the dimensions of the next image in the pyramid
            w = int(image.shape[1] / scale)
            image = imutils.resize(image, width=w)
            print(image.shape)
            # if the resized image does not meet the supplied minimum
            # size, then stop constructing the pyramid
            if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
                break
            # yield the next image in the pyramid
            yield image

    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def max_IoU_search(self, predicted_box, true_boxes):
        IoUs = []
        for i in range(len(true_boxes)):
            IoUs.append(self.bb_intersection_over_union(predicted_box, true_boxes[i]))
        return (np.max(IoUs), np.argmax(IoUs))

    def accuracy(self, y_true, y_pred):
        return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

    def conv_node(self, inputs, filters, kernel_size):
        conv_1 = layers.Conv2D(filters, 1, activation='relu')(inputs)
        conv_2 = layers.SeparableConvolution2D(filters, kernel_size=kernel_size, activation='relu')(conv_1)
        norm_1 = layers.BatchNormalization()(conv_2)
        pool_1 = layers.MaxPooling2D()(norm_1)
        return pool_1

    def make_RPN(self):  # region proposal network
        input_layer = layers.Input(shape=(*self.INPUT_SIZE, 3))

        l1 = self.conv_node(input_layer, 32, 5)
        l2 = self.conv_node(l1, 64, 3)
        l3 = self.conv_node(l2, 64, 3)
        l4 = self.conv_node(l3, 128, 3)

        flat = layers.Flatten()(l4)
        dropout = layers.Dropout(0.2)(flat)
        output_layer = layers.Dense(1, activation='sigmoid')(dropout)

        model = models.Model(input_layer, output_layer)
        model.compile(optimizer='rmsprop', loss='mse', metrics=[self.accuracy])
        model.summary()
        self.RPN = model

    def train_RPN(self, rois, predicted_boxes, true_boxes):
        IoUs = []
        labels = []
        print("\n[INFO] finding max IoU for each RoI")
        start = time.time()
        for i in range(len(predicted_boxes)):
            IoU, label = self.max_IoU_search(predicted_boxes[i], bounding_boxes)
            IoUs.append(IoU)
            labels.append(label)
        end = time.time()
        print("[INFO] searching for max IoUs took {:.5f} seconds".format(end - start))
        print("\n***** Training *****")

        # normalize RoI pixel values
        rois /= 255.
        IoUs = np.array(IoUs, dtype="float32")
        rois, IoUs = shuffle(rois, IoUs)
        self.RPN.fit(rois, IoUs, epochs=100, shuffle=True)
        
    def get_rois(self, orig):
        #(orig_H, orig_W) = orig.shape[:2]
        #orig = imutils.resize(orig, width=rcnn.WIDTH)
        (H, W) = orig.shape[:2]
        #(Delta_H, Delta_W) = (orig_H / H, orig_W / W)

        # initialize the image pyramid
        pyramid = self.image_pyramid(orig, scale=self.PYR_SCALE, minSize=self.MIN_SIZE)
        # initialize two lists, one to hold the ROIs generated from the image
        # pyramid and sliding window, and another list used to store the
        # (x, y)-coordinates of where the ROI was in the original image
        rois = []
        locs = []
        # time how long it takes to loop over the image pyramid layers and
        # sliding window locations
        print("\n[INFO] extracting RoIs from image pyramid")
        start = time.time()
        
        # loop over the image pyramid
        for image in pyramid:
            # determine the scale factor between the *original* image
            # dimensions and the *current* layer of the pyramid
            scale = W / float(image.shape[1])
            # for each layer of the image pyramid, loop over the sliding
            # window locations
            for (x, y, roiOrig) in self.sliding_window(image, self.WIN_STEP, self.ROI_SIZE):
                # scale the (x, y)-coordinates of the ROI with respect to the
                # *original* image dimensions
                x = int(x * scale)
                y = int(y * scale)
                w = int(roiOrig.shape[1] * scale)
                h = int(roiOrig.shape[0] * scale)
                # take the ROI and preprocess it so we can later classify
                # the region using Keras/TensorFlow
                roi = cv2.resize(roiOrig, self.INPUT_SIZE)
                roi = img_to_array(roi)
                roi = preprocess_input(roi)
                # update our list of ROIs and associated coordinates
                rois.append(roi)
                locs.append((x, y, x + w, y + h))

                # check to see if we are visualizing each of the sliding
                # windows in the image pyramid
                if args["visualize"] > 0:
                    # clone the original image and then draw a bounding box
                    # surrounding the current region
                    clone = orig.copy()
                    cv2.rectangle(clone, (x, y), (x + w, y + h),
                        (0, 255, 0), 2)
                    # show the visualization and current ROI
                    cv2.imshow("Visualization", clone)
                    cv2.imshow("ROI", roiOrig)
                    cv2.waitKey(0)

        # show how long it took to loop over the image pyramid layers and
        # sliding window locations
        end = time.time()
        print("[INFO] looping over pyramid/windows took {:.5f} seconds".format(end - start))
        # convert the ROIs to a NumPy array
        rois = np.array(rois, dtype="float32")
        locs = np.array(locs, dtype="int32")
        return (rois, locs)

    def predict(self, rois):
        # classify each of the proposal ROIs using ResNet and then show how
        # long the classifications took
        print("[INFO] classifying ROIs...")
        start = time.time()
        preds = self.RPN.predict(rois)
        print(preds)
        print(preds.shape)
        end = time.time()
        print("[INFO] classifying ROIs took {:.5f} seconds".format(end - start))
        # decode the predictions and initialize a dictionary which maps class
        # labels (keys) to any ROIs associated with that label (values)
        #preds = imagenet_utils.decode_predictions(preds, top=1)
        #labels = {}
        return preds

    def show_pred(self, preds):
        # loop over the predictions
        for (i, p) in enumerate(preds):
            # grab the prediction information for the current ROI
            (imagenetID, label, prob) = p[0]
            # filter out weak detections by ensuring the predicted probability
            # is greater than the minimum probability
            if prob >= args["min_conf"]:
                # grab the bounding box associated with the prediction and
                # convert the coordinates
                box = locs[i]
                # grab the list of predictions for the label and add the
                # bounding box and probability to the list
                L = labels.get(label, [])
                L.append((box, prob))
                labels[label] = L

        # loop over the labels for each of detected objects in the image
        for label in labels.keys():
            # clone the original image so that we can draw on it
            print("[INFO] showing results for '{}'".format(label))
            clone = orig.copy()
            # loop over all bounding boxes for the current label
            for (box, prob) in labels[label]:
                # draw the bounding box on the image
                (startX, startY, endX, endY) = box
                cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
            # show the results *before* applying non-maxima suppression, then
            # clone the image again so we can display the results *after*
            # applying non-maxima suppression
            cv2.imshow("Before", clone)
            clone = orig.copy()

            # extract the bounding boxes and associated prediction
            # probabilities, then apply non-maxima suppression
            boxes = np.array([p[0] for p in labels[label]])
            proba = np.array([p[1] for p in labels[label]])
            boxes = non_max_suppression(boxes, proba)
            # loop over all bounding boxes that were kept after applying
            # non-maxima suppression
            for (startX, startY, endX, endY) in boxes:
                # draw the bounding box and label on the image
                cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            # show the output after apply non-maxima suppression
            cv2.imshow("After", clone)
            cv2.waitKey(0)


if __name__ == "__main__":
    
    args = get_args()
    images, bounding_boxes, labels = get_data('../data/object_detection/') #args["image"])
    
    rcnn = RCNN(args)
    model = ResNet50(weights="imagenet", include_top=True)
    
    rois, box_predictions = rcnn.get_rois(images[0])
    print("\nROIs shape:", rois.shape, "\nBouding box predictions:", box_predictions.shape, "\nBounding boxes:", bounding_boxes.shape)
    
    rcnn.train_RPN(rois, box_predictions, bounding_boxes)
    preds = rcnn.predict(rois)
    #rcnn.show_pred(preds, labels, bounding_boxes)
