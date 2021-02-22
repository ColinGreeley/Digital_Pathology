
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import EfficientNetB2
import sys
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

tile_size = 256
increment_size = tile_size // 1
tissue = "testes"


def make_model(weights=None):
    cb1 = EfficientNetB2(weights='imagenet', include_top=False, drop_connect_rate=0.4, pooling='avg', input_shape=(256, 256, 3))

    x = cb1.output
    x = layers.GaussianNoise(1.0)(x)
    x = layers.Dense(2, activation='softmax', kernel_regularizer='l1_l2')(x)

    model = Model(cb1.input, x)
    model.compile(optimizer=optimizers.Adam(0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if weights:
        model.load_weights(weights)
    return model

def predict(tiles, locs, model):
    preds = model.predict(tiles)
    # return center locations of diseased tiles
    return [locs[i] for i in range(len(preds)) if preds[i, 0] > 0.95]

def rescale(image):
    return cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

def containsWhite(image):
    """Return True if any pixel is white."""
    # Check for white pixels
    lower = (251,251,251)
    upper = (255,255,255)
    wmask = cv2.inRange(image,lower,upper)
    if cv2.countNonZero(wmask) > 0:
        return True
    return False

def containsTooMuchBackground(image):
    """Return True if image contains more than 50% background color (off white)."""
    h,w,c = image.shape
    threshold = int(round(0.7 * h * w))
    lower = (201,201,201)
    upper = (255,255,255)
    bmask = cv2.inRange(image,lower,upper)
    if cv2.countNonZero(bmask) > threshold:
        return True
    return False

def processImage(image_file_name):
    image = cv2.imread(image_file_name, cv2.IMREAD_COLOR)
    height, width, channels = image.shape
    x1 = y1 = 0
    x2 = y2 = tile_size # yes, TILE_SIZE, not (TILE_SIZE - 1)
    tiles = []
    locs = []
    while y2 <= height: # yes, <=, not <
        while x2 <= width:
            tile_image = image[y1:y2, x1:x2]
            if (not containsTooMuchBackground(tile_image)):
                tiles.append(rescale(tile_image))
                locs.append((x1, y1, x2, y2))
            x1 += increment_size
            x2 += increment_size
        x1 = 0
        x2 = tile_size
        y1 += increment_size
        y2 += increment_size
    return np.asarray(tiles), np.asarray(locs), image

def highlight_image(image, locs):
    clone = image.copy()
    (orig_H, orig_W) = clone.shape[:2]
    clone = cv2.resize(clone, (orig_W//10,orig_H//10))
    c1 = clone.copy()
    (H, W) = clone.shape[:2]
    (Delta_H, Delta_W) = (H / orig_H, W / orig_W)
    for x1, y1, x2, y2 in locs:
        #cv2.rectangle(clone, (int(x1*Delta_W), int(y1*Delta_H)), (int(x2*Delta_W), int(y2*Delta_H)), (0, 0, 0))
        clone[int(y1*Delta_H):int(y2*Delta_H), int(x1*Delta_W):int(x2*Delta_W), 0] = 0
        clone[int(y1*Delta_H):int(y2*Delta_H), int(x1*Delta_W):int(x2*Delta_W), 1] = 0
        clone[int(y1*Delta_H):int(y2*Delta_H), int(x1*Delta_W):int(x2*Delta_W), 2] = 0
    res = cv2.addWeighted(clone, 0.2, c1, 0.8, 1.0)
    cv2.imshow("Visualization", res)
    cv2.waitKey(0)
    cv2.imwrite('./highlighted_{}{}_image.jpg'.format(tissue, tile_size), clone)


if __name__ == "__main__":
    image_file = sys.argv[1]                                                                # a. take new image as input
    if len(sys.argv) > 2:
        tissue = sys.argv[2]
    if len(sys.argv) > 3:
        tile_size = int(sys.argv[3])
        increment_size = tile_size // 2
    tiles, locs, image = processImage(image_file)                               # b. generate tiles
    model_weights = 'experiment/{}{}_weights.h5'.format(tissue, tile_size)
    model = make_model(model_weights)                                                       # c. read in trained model
    pred_locs = predict(tiles, locs, model)                                                 # d. classify images

    with open('./pathology_locations.txt', 'w') as f:                                       # e. outputs diseased locations
        f.writelines([str(i) + '\n' for i in pred_locs])
    highlight_image(image, pred_locs)                                                       # f. highlight image with predictions