"""
Colin Greeley
Accuracy Experiment
Used to train tile classifying models and to record the training and testing metrics

Usage: python3 train_tile_classifier.py <tissue_type> <model_size>
"""


import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2, EfficientNetB5
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow.keras.backend as K
from tensorflow.keras import layers, models, Model, optimizers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image
import time
import cv2
import os
import sys
import gc

physical_devices = tf.config.experimental.list_physical_devices('GPU')
#if physical_devices != []:
#    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

image_rez = 256
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, brightness_range=(0.5, 1.5), rotation_range=30, channel_shift_range=100, shear_range=0.2)

def freeze_model(m, block):
    #m.trainable = True
    i = 0
    while True:
        if 'block{}'.format(block) in m.layers[i].name:
            break
        m.layers[i].trainable = False
        i += 1

def make_model():
    cb = EfficientNetB2(weights='imagenet', include_top=False, drop_connect_rate=0.4, pooling='avg', input_shape=(256, 256, 3))
    freeze_model(cb, 4)
    x = cb.output
    #x = layers.GaussianNoise(1.0)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(2, activation='softmax', kernel_regularizer='l1_l2')(x)
    model = Model(cb.input, x)
    #model.summary()
    model.compile(optimizer=optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def get_data(data_dir, validation_size=0.01, test_size=0.01):
    diseased_dir = data_dir + 'diseased/'
    non_diseased_dir = data_dir + 'non_diseased/'

    start = time.time()
    print("\n[INFO] Gathering images and converting them to np arrays")
    diseased_images = np.asarray([cv2.imread(diseased_dir + im_path) for im_path in os.listdir(diseased_dir)])
    non_diseased_images = np.asarray([cv2.imread(non_diseased_dir + im_path) for im_path in os.listdir(non_diseased_dir) if 0.5 > np.random.random()])
    print("[INFO] Conversion took {} seconds".format(round(time.time() - start, 2)))

    #diseased_images = np.array([i for i in diseased_images])
    #non_diseased_images = np.array([i for i in non_diseased_images])
    np.random.shuffle(diseased_images)
    np.random.shuffle(non_diseased_images)

    test_size = int(len(diseased_images) * test_size)
    validation_size = int(len(diseased_images) * validation_size)

    test_diseased_images = diseased_images[-test_size:]
    test_non_diseased_images = non_diseased_images[-test_size:]
    diseased_images = diseased_images[:-test_size]
    non_diseased_images = non_diseased_images[:-test_size]

    validation_diseased_images = diseased_images[-validation_size:]
    validation_non_diseased_images = non_diseased_images[-validation_size:]
    diseased_images = diseased_images[:-validation_size]
    non_diseased_images = non_diseased_images[:-validation_size]

    print("\nDiseased image for training:", diseased_images.shape[0])
    print("Non_diseased image fo training:", non_diseased_images.shape[0])
    print("Diseased image for validation:", validation_diseased_images.shape[0])
    print("Non_diseased image fo validation:", validation_non_diseased_images.shape[0])
    print("Diseased image for testing:", test_diseased_images.shape[0])
    print("Non_diseased image fo testing:", test_non_diseased_images.shape[0])
    print("Total images:", len(diseased_images) + len(non_diseased_images) + len(validation_diseased_images) + len(validation_non_diseased_images) + len(test_diseased_images) + len(test_non_diseased_images))

    #train_data = np.concatenate([diseased_images, non_diseased_images], axis=0)
    #train_labels = np.concatenate([[0 for i in range(len(diseased_images))], [1 for i in range(len(non_diseased_images))]], axis=0)

    validation_data = np.concatenate([validation_diseased_images, validation_non_diseased_images], axis=0)
    validation_labels = np.concatenate([[0 for i in range(len(validation_diseased_images))], [1 for i in range(len(validation_non_diseased_images))]], axis=0)

    test_data = np.concatenate([test_diseased_images, test_non_diseased_images], axis=0)
    test_labels = np.concatenate([[0 for i in range(len(test_diseased_images))], [1 for i in range(len(test_non_diseased_images))]], axis=0)

    # shuffle the data and labels together
    train = (diseased_images, non_diseased_images)
    val = shuffle(validation_data, validation_labels)
    test = shuffle(test_data, test_labels)
    return (train, val, test)

def update_history(h1, h2):
    h1["loss"].extend(h2["loss"])
    h1["accuracy"].extend(h2["accuracy"])
    h1["val_loss"].extend(h2["val_loss"])
    h1["val_accuracy"].extend(h2["val_accuracy"])
    return h1

def experiment_1(data, batch_size, epochs=100, step_size=1, verbose=1):

    if verbose == 1:
        print('\n-------------------\n    Experiment 1\n-------------------')

    (diseased_images, non_diseased_images), (x_val, y_val), (x_test, y_test) = data

    # 5. Train model using TrainingSet and ValidationSet
    model = make_model()
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-5)
    for i in range(0, epochs, step_size):
        nd = non_diseased_images [np.random.choice(len(non_diseased_images), len(diseased_images), replace=False)]
        train_data = np.concatenate([diseased_images, nd], axis=0)
        train_labels = np.concatenate([[0 for i in range(len(diseased_images))], [1 for i in range(len(nd))]], axis=0)
        x_train, y_train = shuffle(train_data, train_labels)
        h = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train)//batch_size, initial_epoch=i, epochs=i+step_size, validation_data=(x_val, y_val), callbacks=[reduce]).history
        history = update_history(history, h)
        tf.keras.backend.clear_session()
        gc.collect()

    # 6. Test model on TestingSet, report accuracy
    result = model.evaluate(x_test, y_test, verbose=0)
    cm = confusion_matrix(y_test, np.argmax(model.predict(x_test), axis=-1))
    if verbose == 1:
        print("Testing loss:", result[0])
        print("Testing accuracy:", result[1])
        print("confusion matrix:\n", cm)

    return result, model, history, cm


if __name__ == "__main__":

    #base_dir = './tiled_datasets_large/'
    base_dir = './tiled_datasets_full/'
    #data_dir = base_dir + 'testes' + '_randomtiles' + '1024' + '/'
    if len(sys.argv) < 3:
        tissue_type = ''
    else:
        tissue_type = sys.argv[2]
    data_dir = base_dir + sys.argv[1] + '_randomtiles' + tissue_type + '/'
    batch_size = 32

    data = get_data(data_dir)

    result, model, history, cm = experiment_1(data, batch_size)
    #experiment_2(data, output_size, batch_size)
    #experiment_3(data, output_size, batch_size)

    model.save_weights("./deployment_models/{}{}_weights.h5".format(sys.argv[1], tissue_type))
    with open("./deployment_models/{}{}.txt".format(sys.argv[1], tissue_type), 'w') as f:
        f.write("Results\n\n")
        f.write("Evaluation loss and accuracy:\n")
        f.write(str(result))
        f.write("\n\nConfusion matrix\n")
        f.write(str(cm))
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