
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionResNetV2
import sys
import cv2
import numpy as np
from generatetiles import processNonDiseasedImage   # importing modified image generation function
                                                    # from original file provided by Larry Holder

image_rez = 256 

def make_feature_extractor():
    conv_base = InceptionResNetV2(weights='imagenet', 
                                include_top=False, 
                                input_shape=(image_rez, image_rez, 3))
    output = layers.GlobalAveragePooling2D()(conv_base.output)
    return Model(conv_base.input, output), output.shape[-1]

def predict(tiles, locs, model):
    preds = model.predict(tiles)
    # return center locations of diseased tiles
    return [((locs[i][0]+locs[i][2])/2, (locs[i][1]+locs[i][3])/2)  for i in range(len(preds)) if np.argmax(preds[i]) == 0] 

def highlight_image(image, locs):
    clone = image.copy()
    (orig_H, orig_W) = clone.shape[:2]
    clone = cv2.resize(clone, (800,1000))
    (H, W) = clone.shape[:2]
    (Delta_H, Delta_W) = (H / orig_H, W / orig_W)
    for x, y in locs:
        #cv2.rectangle(clone, (int(x1*Delta_W), int(y1*Delta_H)), (int(x2*Delta_W), int(y2*Delta_H)), (0, 255, 0), 2)
        cv2.circle(clone, (int(x*Delta_W),int(y*Delta_H)), 20, (0,255,0), 2)
        cv2.circle(clone, (int(x*Delta_W),int(y*Delta_H)), 2, (0,0,255), 2)
        #clone[int(y1*Delta_H):int(y2*Delta_H), int(x1*Delta_W):int(x2*Delta_W), 0] += 50
    cv2.imshow("Visualization", clone)
    cv2.waitKey(0)
    cv2.imwrite('./highlighted_image.jpg', clone)


if __name__ == "__main__":
    origImageFileName = sys.argv[1]                                     # a. take new image as input
    tiles, locs, image = processNonDiseasedImage(origImageFileName)     # b. generate tiles
    model = load_model('tiles256_model.h5')                             # c. read in trained model
    fe, _ = make_feature_extractor()
    pred_locs = predict(fe.predict(tiles/255.), locs, model)            # d. classify images 

    with open('./pathology_locations.txt', 'w') as f:                   # e. outputs diseased locations
        f.writelines([str(i) + '\n' for i in pred_locs])                           
    highlight_image(image, pred_locs)                                   # f. highlight image with predictions