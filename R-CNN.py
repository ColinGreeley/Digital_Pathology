
## Original code from pyimagesearch ##
## https://www.pyimagesearch.com/category/deep-learning/ ##
## "Turning any CNN image classifier into an object detector with Keras, Tensorflow, and OpenCV"  ##
## "Intersection over Union (IoU) for object detection" ##

from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras import models, layers
from tensorflow.keras.models import load_model
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

label_map = {'h':0, 'b':1, 'g':2, 'y':3}

def get_args():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, help="path to the input images")
    ap.add_argument("-s", "--size", type=str, default="(32, 32)", help="ROI size (in pixels)")
    ap.add_argument("-c", "--min-conf", type=float, default=0.7, help="minimum probability to filter weak detections")
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

    bb_array = []
    labels_array = []
    for annotation in annotations:
        labels = []
        bounding_boxes = []
        for cell in annotation:
            labels.append(cell['name'])
            bb = [cell['bndbox'][key] for key in cell['bndbox'].keys()]
            bounding_boxes.append(bb)
        bb_array.append(np.array(bounding_boxes, dtype='int32'))
        labels_array.append(labels)

    return (images, bb_array, labels_array)


class RCNN:

    def __init__(self, args):
        self.WIDTH = 1000
        self.PYR_SCALE = 1.5    # default: 1.5
        self.MIN_SIZE = (300, 300)
        self.ROI_SIZE = eval(args["size"])
        self.WIN_STEP = int(self.ROI_SIZE[0] / 4)
        self.INPUT_SIZE = (256, 256)
        self.make_RPN()
        self.make_classifier()

    def sliding_window(self, image, step, ws):
        # slide a window across the image
        for y in range(0, image.shape[0] - ws[1], step):
            for x in range(0, image.shape[1] - ws[0], step):
                # yield the current window
                yield (x, y, image[y:y + ws[1], x:x + ws[0]])
                """
                if y + ws[1]*2 < image.shape[0]:
                    yield (x, y, image[y:y + ws[1]*2, x:x + ws[0]])
                if x + ws[0]*2 < image.shape[1]:
                    yield (x, y, image[y:y + ws[1], x:x + ws[0]*2])
                """

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


    def get_IoUs(self, rois, predicted_boxes, bounding_boxes):
        IoUs = []
        bb_numbers = []
        print("\n[INFO] finding max IoU for each RoI")
        start = time.time()
        for i in range(len(predicted_boxes)):
            IoU, label_num = self.max_IoU_search(predicted_boxes[i], bounding_boxes)
            IoUs.append(IoU)
            bb_numbers.append(label_num)
        end = time.time()
        print("[INFO] searching for max IoUs took {:.5f} seconds".format(end - start))
        IoUs = np.array(IoUs, dtype="float32")
        return (IoUs, bb_numbers)

    def get_rois(self, orig):
        (orig_H, orig_W) = orig.shape[:2]
        orig = imutils.resize(orig, width=rcnn.WIDTH)
        (H, W) = orig.shape[:2]
        (Delta_H, Delta_W) = (orig_H / H, orig_W / W)

        # initialize the image pyramid
        pyramid = self.image_pyramid(orig, scale=self.PYR_SCALE, minSize=self.MIN_SIZE)
        rois = []
        locs = []
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
                
                roi = cv2.resize(roiOrig, self.ROI_SIZE)
                # update our list of ROIs and associated coordinates
                rois.append(roi)
                locs.append((x*Delta_W, y*Delta_H, (x + w)*Delta_W, (y + h)*Delta_H))

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

    def get_rois_list(self, images):
        rois_array = []
        box_predictions_array = []
        for image in images:
            rois, box_predictions = rcnn.get_rois(image)
            rois_array.append(rois)
            box_predictions_array.append(box_predictions)
        print('\n' + str(len(rois_array)) + " images")
        print("ROIs shape:", np.asarray(rois_array[0]).shape, 
            "\nBouding box predictions:", np.asarray(box_predictions_array[0]).shape, 
            "\nBounding boxes:", np.asarray(bounding_boxes_array[0]).shape)
        return rois_array, box_predictions_array


    def accuracy(self, y_true, y_pred):
        return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

    def conv_node(self, inputs, filters, kernel_size):
        conv_1 = layers.Conv2D(filters, 1, activation='relu')(inputs)
        conv_2 = layers.Conv2D(filters, kernel_size=kernel_size, padding='same')(conv_1)
        norm_1 = layers.BatchNormalization()(conv_2)
        act_1 = layers.Activation('relu')(norm_1)
        pool_1 = layers.MaxPooling2D()(act_1)
        return pool_1

    def make_RPN(self):  
        # region proposal network
        input_layer = layers.Input(shape=(*self.ROI_SIZE, 3))

        l1 = self.conv_node(input_layer, 32, 3)
        l2 =          self.conv_node(l1, 64, 3)
        l3 =          self.conv_node(l2, 128, 3)

        dropout = layers.Dropout(0.2)(l3)
        flatten = layers.Flatten()(dropout)
        dense_1 = layers.Dense(128, activation='relu')(flatten)
        output_layer = layers.Dense(1, activation='sigmoid')(dense_1)

        model = models.Model(input_layer, output_layer)
        model.compile(optimizer='rmsprop', loss='mse', metrics=[self.accuracy])
        model.summary()
        self.RPN = model

    def make_classifier(self):
        conv_base = InceptionResNetV2(weights='imagenet', 
                                include_top=False, 
                                input_shape=(299, 299, 3))
        for layer in conv_base.layers:
            layer.trainable = False
        pooling = layers.GlobalAveragePooling2D()(conv_base.output)
        dropout = layers.Dropout(0.2)(pooling)
        output = layers.Dense(4, activation='sigmoid')(dropout)

        model = models.Model(conv_base.input, output)
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.classifier = model


    def train_RPN(self, rois_array, box_predictions_array, true_bounding_boxes_array):

        IoUs_array = []
        label_nums_array = []
        for i in range(len(rois_array)):
            IoUs, bb_numbers = rcnn.get_IoUs(rois_array[i], box_predictions_array[i], true_bounding_boxes_array[i])
            IoUs_array.append(IoUs)
            label_nums_array.append(bb_numbers)
        
        x = np.append(*rois_array, axis=0)
        y = np.append(*IoUs_array, axis=0)
        print("X:", x.shape, " Y:", y.shape)

        print("\n***** Training *****")
        # normalize RoI pixel values
        x /= 255.
        x, y = shuffle(x, y)
        self.RPN.fit(x, y, epochs=30, shuffle=True)
        return label_nums_array

    def get_RPN_pred_list(self, rois_array, box_predictions_array, training_labels, confidence):
        probs_array = []
        boxes_array = []
        label_array = []
        for i in range(len(images)):
            probs, boxes, labels = rcnn.RPN_predict(rois_array[i], box_predictions_array[i], training_labels[i], confidence=confidence)
            probs_array.append(probs)
            boxes_array.append(boxes)
            label_array.append(labels)
        return probs_array, boxes_array, label_array

    def train_classifier(self, images, probs_array, boxes_array, label_array):
        
        x = []
        for image, boxes in zip(images, boxes_array):
            x.append([cv2.resize(image[y1:y2, x1:x2], (299, 299)) / 255. for (x1, y1, x2, y2) in boxes])
        
        x = np.append(*x, axis=0)
        y = np.append(*label_array, axis=0)
        print("X:", x.shape, " Y:", y.shape)

        x, y = shuffle(x, y)
        self.classifier.fit(x, y, epochs=30, shuffle=True)

    def RPN_predict(self, rois, box_predictions, labels_array, confidence):
        print("[INFO] classifying ROIs...")
        start = time.time()
        preds = self.RPN.predict(rois/255.)
        end = time.time()
        print("[INFO] classifying ROIs took {:.5f} seconds".format(end - start))
        
        boxes = []
        probs = []
        labels = []
        print(image.shape)
        print(preds.shape)
        print(box_predictions.shape)
        for i in range(len(preds)):
            if preds[i] >= confidence:
                #print(preds[i])
                probs.append(preds[i])
                boxes.append(box_predictions[i])
                labels.append(labels_array[i])
        return probs, boxes, labels

    def classifier_predict(self, image, probs, boxes):
        boxes = non_max_suppression(boxes, probs=probs[:,0], overlapThresh=0.7)
        x = np.array([cv2.resize(image[y1:y2, x1:x2], (299, 299)) / 255. for (x1, y1, x2, y2) in boxes])
        print('classifier prediction input shape:', x.shape)
        preds = self.classifier.predict(x)
        return preds, boxes
        

    def show_RPN_pred(self, image, probs, boxes):
       
        orig = image
        (orig_H, orig_W) = orig.shape[:2]
        orig = imutils.resize(orig, 1000)
        (H, W) = orig.shape[:2]
        (Delta_H, Delta_W) = (orig_H / H, orig_W / W)
        clone = orig.copy()

        boxes = non_max_suppression(boxes, probs=probs[:,0], overlapThresh=0.5)
        for box in boxes:
            (startX, startY, endX, endY) = box
            cv2.rectangle(clone, (int(startX/Delta_W), int(startY/Delta_H)), (int(endX/Delta_W), int(endY/Delta_H)), (0, 0, 0), 1)
        cv2.imshow("Before", clone)
        cv2.waitKey(0)
        return clone

    def show_classifier_pred(self, image, preds, boxes):
        
        orig = image
        (orig_H, orig_W) = orig.shape[:2]
        orig = imutils.resize(orig, 1000)
        (H, W) = orig.shape[:2]
        (Delta_H, Delta_W) = (orig_H / H, orig_W / W)
        clone = orig.copy()

        boxes = non_max_suppression(boxes, probs=probs[:,0], overlapThresh=0.5)
        for box in boxes:
            (startX, startY, endX, endY) = box
            cv2.rectangle(clone, (int(startX/Delta_W), int(startY/Delta_H)), (int(endX/Delta_W), int(endY/Delta_H)), (0, 0, 0), 1)
        cv2.imshow("Before", clone)
        cv2.waitKey(0)


if __name__ == "__main__":
    
    args = get_args()
    images, bounding_boxes_array, labels_array = get_data('../data/object_detection/')
    
    rcnn = RCNN(args)
    
    # get rois for all images
    rois_array, box_predictions_array = rcnn.get_rois_list(images)
    # train RPN
    label_nums_array = rcnn.train_RPN(rois_array, box_predictions_array, bounding_boxes_array)
    # map label numbers to classifier labels
    training_labels = [[label_map[labels_array[i][label_num]] for label_num in label_nums_array[i]] for i in range(len(label_nums_array))]

    #rcnn.RPN = load_model('RPN.h5')

    # get RPN predictions for classification training
    probs_array, boxes_array, label_array = rcnn.get_RPN_pred_list(rois_array, box_predictions_array, training_labels, confidence=0.5)
    # train the classifier
    rcnn.train_classifier(images, probs_array, boxes_array, label_array)

    #rcnn.show_pred(images[0], labels, preds, box_predictions_array[0], confidence=0.5)
