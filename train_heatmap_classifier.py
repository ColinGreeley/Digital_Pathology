
import os
import sys
import numpy as np
import cv2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc
import time
import pickle

TILE_SIZE = 256
TILE_INCREMENT = TILE_SIZE // 2


# ----- Tile Image -----

def containsWhite(image):
    """Return True if any pixel is white."""
    # Check for white pixels
    lower = (251,251,251)
    upper = (255,255,255)
    wmask = cv2.inRange(image,lower,upper)
    if cv2.countNonZero(wmask) > 0:
        return True
    return False

def rescale(image):
    return cv2.resize(image, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_AREA)

def containsTooMuchBackground(image):
    """Return True if image contains more than 70% background color (off white)."""
    h,w,c = image.shape
    threshold = int(round(0.7 * h * w))
    lower = (201,201,201)
    upper = (255,255,255)
    bmask = cv2.inRange(image,lower,upper)
    if cv2.countNonZero(bmask) > threshold:
        return True
    return False


# ----- Classify Image -----

def make_model(weights=None):
    cb = EfficientNetB2(weights='imagenet', include_top=False, drop_connect_rate=0.4, pooling='avg', input_shape=(256, 256, 3))
    x = cb.output
    x = layers.GaussianDropout(0.3)(x)
    x = layers.Dense(2, activation='softmax', kernel_regularizer='l1_l2')(x)
    model = Model(cb.input, x)
    #model.summary()
    model.compile(optimizer=optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if weights:
        model.load_weights(weights)
    return model

def predict(data, model):
    #print("\n[INFO] Predicting image tiles")
    #start = time.time()
    tiles, locs, image = data
    preds = model.predict(tiles)
    # return center locations of diseased tiles
    pred_locs = [[locs[i], preds[i, 0]] for i in range(len(preds)) if preds[i, 0] > 0.5]
    #print("[INFO] Predicting image tiles took {} seconds".format(round(time.time() - start, 2)))
    return (pred_locs, image)

def make_canvas(pred_locs, image):
    clone = image
    canvas = np.zeros((clone.shape[0], clone.shape[1]))
    count = 0
    for ((x1, y1, x2, y2), pred) in pred_locs:
        canvas[y1:y2, x1:x2] += pred
        count += pred
    if np.max(canvas) > 1:
        canvas = canvas / np.max(canvas)
    #clone = cv2.addWeighted(clone, 0.2, clone1, 0.8, 1.0)
    return canvas, count


# ----- Heat maps -----

def processImage(image_file_name):
    image = cv2.imread(image_file_name, cv2.IMREAD_COLOR)
    height, width, channels = image.shape
    x1 = y1 = 0
    x2 = y2 = TILE_SIZE # yes, TILE_SIZE, not (TILE_SIZE - 1)
    tiles = []
    locs = []
    while y2 <= height: # yes, <=, not <
        while x2 <= width:
            tile_image = image[y1:y2, x1:x2]
            if (not containsTooMuchBackground(tile_image) and not containsWhite(tile_image)):
                tiles.append(rescale(tile_image))
                locs.append((x1, y1, x2, y2))
            x1 += TILE_INCREMENT
            x2 += TILE_INCREMENT
        x1 = 0
        x2 = TILE_SIZE
        y1 += TILE_INCREMENT
        y2 += TILE_INCREMENT
    return (np.asarray(tiles), np.asarray(locs), image)

def make_heatmaps(path, model, tissue_type):
    diseased_dir = path + 'images/' + tissue_type + '/diseased/'
    non_diseased_dir = path + 'images/' + tissue_type + '/non_diseased/'
    diseased_heatmap_dir = path + 'heatmaps/' + tissue_type + '/diseased/'
    non_diseased_heatmap_dir = path + 'heatmaps/' + tissue_type + '/non_diseased/'
    diseased_list = os.listdir(diseased_dir)
    non_diseased_list = os.listdir(non_diseased_dir)

    print("\n[INFO] Generating heatmaps for diseased images")
    start = time.time()
    for i, im in enumerate(diseased_list[:1]):
        print(im)
        processed_image = processImage(diseased_dir + im)
        pred_locs, image = predict(processed_image, model)
        canvas, canvas_val = make_canvas(pred_locs, image)
        cv2.imwrite(diseased_heatmap_dir + "diseased_heatmap_{}.jpg".format(i+1), canvas*255)
        with open(diseased_heatmap_dir + "diseased_heatmap_{}.txt".format(i+1), 'w') as f:
            f.write(str(canvas_val))
    print("[INFO] Generating heatmaps for diseased images took {} seconds".format(round(time.time() - start, 2)))
    print("\n[INFO] Generating heatmaps for non_diseased images")
    start = time.time()
    for i, im in enumerate(non_diseased_list[:]):
        print(im)
        processed_image = processImage(non_diseased_dir + im)
        if processed_image[0].size != 0:
            pred_locs, image = predict(processed_image, model)
            canvas, canvas_val = make_canvas(pred_locs, image)
            cv2.imwrite(non_diseased_heatmap_dir + "non_diseased_heatmap_{}.jpg".format(i+1), canvas*255)
            with open(non_diseased_heatmap_dir + "non_diseased_heatmap_{}.txt".format(i+1), 'w') as f:
                f.write(str(canvas_val))
    print("[INFO] Generating heatmaps for non_diseased images took {} seconds".format(round(time.time() - start, 2)))
    #data = np.concatenate([diseased_processed_images, non_diseased_processed_images], axis=0)
    #labels = np.concatenate([[0 for i in range(len(diseased_processed_images))], [1 for i in range(len(non_diseased_processed_images))]], axis=0)
    #return data, labels

def get_heatmap_data(path, tissue_type):
    diseased_heatmap_dir = path + 'heatmaps/' + tissue_type + '/diseased/'
    non_diseased_heatmap_dir = path + 'heatmaps/' + tissue_type + '/non_diseased/'
    diseased_list = os.listdir(diseased_heatmap_dir)
    non_diseased_list = os.listdir(non_diseased_heatmap_dir)
    diseased_heatmaps = []
    diseased_heatmaps_vals = []
    non_diseased_heatmaps = []
    non_diseased_heatmaps_vals = []
    print("\n[INFO] Getting heatmap data")
    start = time.time()
    for i in diseased_list:
        #diseased_heatmap = cv2.imread(diseased_heatmap_dir + im, cv2.IMREAD_UNCHANGED) / 255
        #diseased_heatmaps_vals.append(np.mean(diseased_heatmap))
        if "txt" in i:
            with open(diseased_heatmap_dir + i, 'r') as f:
                diseased_heatmaps_vals.append(float(f.read()))
    for i in non_diseased_list:
        #non_diseased_heatmap = cv2.imread(non_diseased_heatmap_dir + im, cv2.IMREAD_UNCHANGED) / 255
        #non_diseased_heatmaps_vals.append(np.mean(non_diseased_heatmap))
        if "txt" in i:
            with open(non_diseased_heatmap_dir + i, 'r') as f:
                non_diseased_heatmaps_vals.append(float(f.read()))
    print("[INFO] Getting heatmap data took {} seconds".format(round(time.time() - start, 2)))
    return (diseased_heatmaps_vals, non_diseased_heatmaps_vals)

def make_training_data(d_vals, nd_vals):
    data = np.concatenate([d_vals, nd_vals], axis=0)
    labels = np.concatenate([[0 for i in range(len(d_vals))], [1 for i in range(len(nd_vals))]], axis=0)
    data, labels = shuffle(data, labels)
    return data.reshape(-1, 1), labels


# ----- Logistic Regression model -----
def run(X, y, folder, tissue_type):
    kf = KFold(3)
    models = []
    cms = []
    scores = []
    aucs = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        log = LogisticRegression(C=1).fit(X_train, y_train)
        scores.append(log.score(X_test, y_test))
        cms.append(confusion_matrix(y_test, log.predict(X_test)))
        fpr, tpr, _ = roc_curve(1-y_test, log.predict_proba(X_test)[:, 0])
        roc_auc = roc_auc_score(1-y_test, log.predict_proba(X_test)[:, 0])
        plt.plot(fpr, tpr)
        aucs.append(roc_auc)
    plt.plot(np.arange(len(y_test))/len(y_test), np.arange(len(y_test))/len(y_test), linestyle='--')
    plt.title("Average AUC: " + str(round(np.mean(aucs), 4)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(folder + '/' + tissue_type + '_Log_Roc_Curve.png')
    #plt.show()
    with open(folder + '/' + tissue_type +'_log_model.pkl', 'wb') as f:
        pickle.dump(log, f)

    with open("./deployment_models/{}_Log_Model.txt".format(tissue_type), 'w') as f:
        f.write("Results\n\n")
        f.write("Evaluation loss and accuracy:\n")
        f.write(str(scores))
        f.write("\n\nConfusion matrix\n")
        f.write(str(cms))
        f.write("\n\n***History***\n")
        f.write("Loss\n")
        f.write(str(history["loss"]))
        f.write("\n\nAccuracy\n")
        f.write(str(history["accuracy"]))
        f.write("\n\nVal_loss\n")
        f.write(str(history["val_loss"]))
        f.write("\n\nVal_accuracy\n")
        f.write(str(history["val_accuracy"]))
        f.write('\n\n')


if __name__ == "__main__":
    data_path = '../data/research/'
    tissue_type = sys.argv[1]
    if True:
        model = make_model('experiment3/{}_weights.h5'.format(tissue_type))
        make_heatmaps(data_path, model, tissue_type)
    (diseased_heatmaps_vals, non_diseased_heatmaps_vals) = get_heatmap_data(data_path, tissue_type)
    training_data, training_labels = make_training_data(diseased_heatmaps_vals, non_diseased_heatmaps_vals)
    run(training_data, training_labels, 'deployment_models/', tissue_type)
