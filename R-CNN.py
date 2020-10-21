
## Colin Greeley ##
## Original code from pyimagesearch ##
## https://www.pyimagesearch.com/category/deep-learning/ ##
## "Turning any CNN image classifier into an object detector with Keras, Tensorflow, and OpenCV"  ##
## "Intersection over Union (IoU) for object detection" ##

from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras import models, layers
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from imutils.object_detection import non_max_suppression
from generatetiles import containsWhite
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
value_map = ['healthy', 'blue', 'green', 'yellow']
color_map = [(0,0,0), (255,0,0), (0,255,0), (0,255,255)] #BGR

def get_args():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", type=str, required=True, help='-t "test" or "train_all" or "train_rpn" or "train_classifier"')
    ap.add_argument("-p", "--path", type=str, required=True, help="path to test image (test) or path to dataset (train)")
    ap.add_argument("-s", "--size", type=str, default="(32, 32)", help="ROI size (in pixels)")
    ap.add_argument("-c", "--min-conf", type=float, default=0.3, help="minimum probability to filter weak detections")
    ap.add_argument("-v", "--visualize", type=int, default=-1, help="whether or not to show extra visualizations for debugging")
    args = vars(ap.parse_args())
    return args

def get_data(path, stop=10000):
    if path[-1] != '/':
        path += '/'
    images_path = path + 'JPEGImages/'
    annotations_path = path + 'Annotations/'

    images = []
    for i, im in enumerate(os.listdir(images_path)):
        if i >= stop:
            break
        images.append(cv2.imread(images_path + im))
        
    annotations = []
    for i, an in enumerate(os.listdir(annotations_path)):
        if i >= stop:
            break
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


class RPN:

    def __init__(self, t):
        self.WIDTH = 1200
        self.PYR_SCALE = 1.3    # default: 1.5
        self.MIN_SIZE = (300, 300)
        self.ROI_SIZE = (32, 32)
        self.WIN_STEP = int(self.ROI_SIZE[0] / 4)
        if t == 'test':
            self.model = None
        else:
            self.make_model()
        self.datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, brightness_range=(-0.2,0.2))

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
        while True:
            # compute the dimensions of the next image in the pyramid
            w = int(image.shape[1] / scale)
            image = imutils.resize(image, width=w)
            #print(image.shape)
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
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def max_IoU_search(self, predicted_box, true_boxes):
        IoUs = []
        for i in range(len(true_boxes)):
            IoUs.append(self.bb_intersection_over_union(predicted_box, true_boxes[i]))
        return (np.max(IoUs), np.argmax(IoUs))


    def get_IoUs(self, predicted_boxes, bounding_boxes, imn):
        IoUs = []
        bb_numbers = []
        print("\n[INFO] Finding max IoU for each RoI in image", imn)
        start = time.time()
        for i in range(len(predicted_boxes)):
            IoU, label_num = self.max_IoU_search(predicted_boxes[i], bounding_boxes)
            IoUs.append(IoU)
            bb_numbers.append(label_num)
        end = time.time()
        print("[INFO] Searching for max IoUs took {:.5f} seconds".format(end - start))
        IoUs = np.array(IoUs, dtype="float32")
        return (IoUs, bb_numbers)

    def get_rois(self, orig):
        (orig_H, orig_W) = orig.shape[:2]
        orig = imutils.resize(orig, width=self.WIDTH)
        (H, W) = orig.shape[:2]
        (Delta_H, Delta_W) = (orig_H / H, orig_W / W)

        pyramid = self.image_pyramid(orig, scale=self.PYR_SCALE, minSize=self.MIN_SIZE)
        rois = []
        locs = []
        print("\n[INFO] Extracting RoIs from image pyramid")
        start = time.time()
        
        for image in pyramid:
            scale = W / float(image.shape[1])
            for (x, y, roiOrig) in self.sliding_window(image, self.WIN_STEP, self.ROI_SIZE):
                # scale the (x, y)-coordinates of the ROI with respect to the
                # *original* image dimensions
                x = int(x * scale)
                y = int(y * scale)
                w = int(roiOrig.shape[1] * scale)
                h = int(roiOrig.shape[0] * scale)
                
                roi = cv2.resize(roiOrig, self.ROI_SIZE)
                # update our list of ROIs and associated coordinates
                if not containsWhite(roiOrig):
                    rois.append(roi)
                    locs.append((x*Delta_W, y*Delta_H, (x + w)*Delta_W, (y + h)*Delta_H))

                # check to see if we are visualizing each of the sliding
                # windows in the image pyramid
                if args["visualize"] > 0:
                    # clone the original image and then draw a bounding box
                    # surrounding the current region
                    clone = orig.copy()
                    cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # show the visualization and current ROI
                    cv2.imshow("Visualization", clone)
                    cv2.imshow("ROI", roiOrig)
                    cv2.waitKey(0)

        end = time.time()
        print("[INFO] looping over pyramid/windows took {:.5f} seconds".format(end - start))
        # convert the ROIs to a NumPy array
        rois = np.array(rois, dtype="float32")
        locs = np.array(locs, dtype="int32")
        return (rois, locs)

    def get_rois_list(self, images, get_images=False):
        if get_images == False:
            rois_array = []
            for image in images:
                _, rois = self.get_rois(image)
                rois_array.append(rois)
            print('\n' + str(len(rois_array)) + " images")
            return ([], rois_array)
        else:
            roi_images_array = []
            rois_array = []
            for image in images:
                roi_images, rois = self.get_rois(image)
                roi_images_array.append(roi_images)
                rois_array.append(rois)
            print('\n' + str(len(rois_array)) + " images")
            return (roi_images_array, rois_array)

    def get_label_data(self, box_predictions_array, true_bounding_boxes_array):
        IoUs_array = []
        label_nums_array = []
        for i in range(len(box_predictions_array)):
            IoUs, bb_numbers = self.get_IoUs(box_predictions_array[i], true_bounding_boxes_array[i], i+1)
            IoUs_array.append(IoUs)
            label_nums_array.append(bb_numbers)
        return IoUs_array, label_nums_array

    def get_classifier_data(self, box_predictions, IoUs_array, labels_array, confidence):
        b = []
        l = []
        for i in range(len(IoUs_array)):
            if IoUs_array[i] > confidence:
                b.append(box_predictions[i])
                l.append(labels_array[i])
        return b, l


    def accuracy(self, y_true, y_pred):
        return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

    def conv_node(self, inputs, filters, kernel):
        conv_a1 = layers.Conv2D(filters, 1, activation='relu')(inputs)
        conv_a2 = layers.Conv2D(filters, kernel, padding='same')(conv_a1)
        norm_a = layers.BatchNormalization()(conv_a2)
        act_a = layers.Activation('relu')(norm_a)
        pool_a = layers.MaxPooling2D()(act_a)
        return pool_a

    def make_model(self):  
        # region proposal network
        input_layer = layers.Input(shape=(*self.ROI_SIZE, 3))

        l1 = self.conv_node(input_layer, 32, 3)
        l2 =          self.conv_node(l1, 64, 3)
        l3 =          self.conv_node(l2, 128, 3)

        flatten = layers.Flatten()(l3)
        dense_1 = layers.Dense(256, activation='relu')(flatten)
        dropout_1 = layers.Dropout(0.3)(dense_1)
        dense_2 = layers.Dense(64, activation='relu')(dropout_1)
        dropout_2 = layers.Dropout(0.2)(dense_2)
        output_layer = layers.Dense(1, activation='sigmoid')(dropout_2)

        model = models.Model(input_layer, output_layer)
        model.compile(optimizer='rmsprop', loss='mse', metrics=[self.accuracy])
        #model.summary()
        self.model = model


    def train(self, rois_array, IoUs_array):

        x = np.concatenate(rois_array, axis=0)
        y = np.concatenate(IoUs_array, axis=0)
        print("X:", x.shape, " Y:", y.shape)

        # normalize RoI pixel values
        x /= 255.
        x, y = shuffle(x, y)
        self.model.fit(self.datagen.flow(x, y, batch_size=32), steps_per_epoch=len(x)//32, epochs=50)
        self.model.save("RPN.h5")

    def predict(self, roi_images, rois, labels_array=None, confidence=0.3):

        print("\n[INFO] Classifying ROIs...")
        start = time.time()
        preds = self.model.predict(roi_images/255.)
        end = time.time()
        print("[INFO] Classifying ROIs took {:.5f} seconds".format(end - start))
        
        boxes = []
        probs = []
        labels = []
        print("Preds shape:", preds.shape)
        print("Box predictions shape:", rois.shape)
        for i in range(len(preds)):
            if preds[i] >= confidence:
                #print(preds[i])
                probs.append(preds[i])
                boxes.append(rois[i])
                if labels_array is not None:
                    labels.append(labels_array[i])
        return np.array(probs), np.array(boxes), np.array(labels)

    def get_pred_list(self, rois_array, IoUs_array, training_labels, confidence):
        boxes_array = []
        label_array = []
        for i in range(len(rois_array)):
            #probs, boxes, labels = self.predict(roi_images_array[i], rois_array[i], training_labels[i], confidence=confidence)
            boxes, labels = self.get_classifier_data(rois_array[i], IoUs_array[i], training_labels[i], confidence)
            boxes_array.append(boxes)
            label_array.append(labels)
        return boxes_array, label_array

    def show_pred(self, image, probs, boxes, thresh):
       
        orig = image
        (orig_H, orig_W) = orig.shape[:2]
        orig = imutils.resize(orig, 1000)
        (H, W) = orig.shape[:2]
        (Delta_H, Delta_W) = (orig_H / H, orig_W / W)
        clone = orig.copy()

        boxes = non_max_suppression(boxes, probs=probs[:,0], overlapThresh=thresh)
        for box in boxes:
            (startX, startY, endX, endY) = box
            cv2.rectangle(clone, (int(startX/Delta_W), int(startY/Delta_H)), (int(endX/Delta_W), int(endY/Delta_H)), (0, 0, 0), 1)
        cv2.imshow("Before", clone)
        cv2.waitKey(0)
        cv2.imwrite('./RPN_Prediction.jpg', clone)
        return clone


class Classifier:

    def __init__(self, t):
        if t == 'test':
            self.model = None
        else:
            self.make_model()
        self.datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, brightness_range=(-0.2,0.2))
    
    def make_model(self):
        conv_base = InceptionResNetV2(weights='imagenet', 
                                include_top=False, 
                                input_shape=(299, 299, 3))
        for layer in conv_base.layers:
            layer.trainable = False
        pooling = layers.GlobalAveragePooling2D()(conv_base.output)
        dropout_1 = layers.Dropout(0.3)(pooling)
        dense = layers.Dense(256, activation='relu')(dropout_1)
        dropout_2 = layers.Dropout(0.2)(dense)
        output = layers.Dense(4, activation='sigmoid')(dropout_2)

        model = models.Model(conv_base.input, output)
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def custom_non_max_suppression(self, boxes, probs=None, overlapThresh=0.3):
        if len(boxes) == 0:
            return []
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = y2
        if probs is not None:
            idxs = probs
        idxs = np.argsort(idxs)
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))
        return pick

    def train(self, images, ba, la):

        boxes_array = ba
        label_array = la
        # remove 50% of the heathly cells from training set
        for i in range(len(boxes_array)):
            j = 0
            while j < len(label_array[i]):
                if label_array[i][j] == 0 and np.random.random() < 0.8:
                    boxes_array[i] = np.delete(boxes_array[i], j, axis=0)
                    label_array[i] = np.delete(label_array[i], j)
                else:
                    j += 1

        # extract images from bounding box coordinates
        print("\n[INFO] Extracting images from bounding box coordinates...")
        x = []
        for image, boxes in zip(images, boxes_array):
            x.append([cv2.resize(image[y1:y2, x1:x2], (299, 299)) / 255. for (x1, y1, x2, y2) in boxes])
        
        x = np.concatenate(x, axis=0)
        y = np.concatenate(label_array, axis=0)

        print("X:", x.shape, " Y:", y.shape)
        print("[INFO] Training classifier...")

        x, y = shuffle(x, y)
        self.model.fit(self.datagen.flow(x, y, batch_size=32), steps_per_epoch=len(x)//32, epochs=500)
        self.model.save("classifier.h5")

    def predict(self, image, probs, boxes, thresh):
        boxes = non_max_suppression(boxes, probs=probs[:,0], overlapThresh=thresh)
        x = np.array([cv2.resize(image[y1:y2, x1:x2], (299, 299)) / 255. for (x1, y1, x2, y2) in boxes])
        print('\nClassifier prediction input shape:', x.shape)
        print("[INFO] Classifying pathologies...")
        start = time.time()
        preds = self.model.predict(x)
        end = time.time()
        print("[INFO] Classifying pathologies took {:.5f} seconds".format(end - start))
        return preds, boxes
    
    def show_pred(self, image, preds, boxes, confidence):
        
        orig = image
        (orig_H, orig_W) = orig.shape[:2]
        orig = imutils.resize(orig, 1000)
        (H, W) = orig.shape[:2]
        (Delta_H, Delta_W) = (orig_H / H, orig_W / W)
        clone = orig.copy()

        diseased_boxes = []
        diseased_preds = []
        for i in range(len(preds)):
            if preds[i, 1] >= confidence or preds[i, 2] >= confidence or preds[i, 3] >= confidence:
                diseased_boxes.append(boxes[i])
                diseased_preds.append(preds[i, 1:])
        diseased_boxes = np.array(diseased_boxes)
        diseased_preds = np.array(diseased_preds)

        idxs = self.custom_non_max_suppression(diseased_boxes, probs=np.max(diseased_preds, axis=-1), overlapThresh=0.3)
        for box, pred in zip(diseased_boxes[idxs], diseased_preds[idxs]):
            (startX, startY, endX, endY) = box
            cv2.rectangle(clone, (int(startX/Delta_W), int(startY/Delta_H)), (int(endX/Delta_W), int(endY/Delta_H)), color_map[np.argmax(pred)+1], 1)
            cv2.putText(clone, '{:.0f}%'.format(max(pred)*100), (int(startX/Delta_W), int(startY/Delta_H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        cv2.imshow("Before", clone)
        cv2.waitKey(0)
        cv2.imwrite('./Classifier_Prediction.jpg', clone)
  

class RCNN:

    def __init__(self, args):
        self.args = args
        self.rpn = RPN(args['type'])
        self.classifier = Classifier(args['type'])
        self.fully_annotated_images = 5

    def __call__(self, data_path, instruction):
        if instruction == 'train_all':
            self.train_all(data_path)
        if instruction == 'train_rpn':
            self.train_rpn(data_path)
        if instruction == 'train_classifier':
            self.train_classifier(data_path)
        if instruction == 'test':
            self.test(data_path)

    def train_rpn(self, data_path):
        # get images and respective XML data
        images, bounding_boxes_array, _ = get_data(data_path, self.fully_annotated_images)
        # get rois for all images
        roi_images_array, rois_array = self.rpn.get_rois_list(images, True)
        # get label data
        IoUs_array, _ = self.rpn.get_label_data(rois_array, bounding_boxes_array)
        # train RPN
        self.rpn.train(roi_images_array, IoUs_array)

    def train_classifier(self, data_path):
        # get images and respective XML data
        images, bounding_boxes_array, labels_array = get_data(data_path)
        # get rois for all images
        _, rois_array = self.rpn.get_rois_list(images)
        # get label data
        IoUs_array, label_nums_array = self.rpn.get_label_data(rois_array, bounding_boxes_array)
        # map label numbers to classifier labels
        training_labels = [[label_map[labels_array[i][label_num]] for label_num in label_nums_array[i]] for i in range(len(label_nums_array))]
        # get RPN predictions for classification training
        boxes_array, label_array = self.rpn.get_pred_list(rois_array, IoUs_array, training_labels, confidence=0.3)
        # train the classifier
        self.classifier.train(images, boxes_array, label_array)

    def train_all(self, data_path):
        self.train_rpn(data_path)
        self.train_classifier(data_path)

    def test(self, data_path):
        image = cv2.imread(data_path)
        # load pre-trained model
        self.rpn.model = load_model('RPN.h5')
        self.classifier.model = load_model('classifier.h5')
        # get rois
        roi_images, rois = self.rpn.get_rois(image)
        # rpn predictions
        probs, RPN_boxes, _ = self.rpn.predict(roi_images, rois, confidence=0.3)   # confidence of bounding box being over cell
        # classifier predictions
        preds, boxes = self.classifier.predict(image, probs, RPN_boxes, thresh=0.5)     # non-max suppression threshold (higher number allows more overlap)
        # show output                                                                    
        self.rpn.show_pred(image, probs, RPN_boxes, thresh=0.5)                         # non-max suppression threshold
        self.classifier.show_pred(image, preds, boxes, confidence=0.7)                  # confidence of pathology classification



if __name__ == "__main__":
    
    args = get_args()
    
    RCNN(args)(args['path'], args['type'])