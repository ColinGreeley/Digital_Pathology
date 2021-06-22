"""
Colin Greeley
Accuracy Experiment
Used to train tile classifying models and to record the training and testing metrics

Usage: python3 train_tile_classifier.py <tissue_type> <model_size>
"""

#from astunparse.unparser import main
#import silence_tensorflow.auto
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2, EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.metrics import AUC
import tensorflow.keras.backend as K
from tensorflow.keras import layers, models, Model, optimizers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle, class_weight
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc, classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2
import os
import sys
import gc

from tensorflow.python.keras import regularizers

physical_devices = tf.config.experimental.list_physical_devices('GPU')
#if physical_devices != []:
#    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

image_rez = 256
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=30, channel_shift_range=150, shear_range=0.2, validation_split=0.2)

def freeze_model(m, block):
    #m.trainable = True
    i = 0
    while i < len(m.layers):
        if 'block{}'.format(block) in m.layers[i].name:
            break
        m.layers[i].trainable = False
        i += 1
    while i < len(m.layers):
        if isinstance(m.layers[i], layers.BatchNormalization):
            m.layers[i].trainable = False
        i += 1

def unfreeze_model(model):
    #m.trainable = True
    i = 0
    while i < len(model.layers):
        if not isinstance(model.layers[i], layers.BatchNormalization):
            model.layers[i].trainable = True
        i += 1
    model.compile(optimizer=optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def make_model(num_classes):
    reg = regularizers.l1_l2(0.01, 0.01)
    cb = EfficientNetB2(weights='imagenet', include_top=False, drop_connect_rate=0.4, pooling='avg', input_shape=(256, 256, 3))
    #cb.trainable = False
    freeze_model(cb, 3)
    x = cb.output
    #x = layers.GaussianNoise(0.5)(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.3)(x)
    #x = layers.Dense(128, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(num_classes, activation='softmax', kernel_regularizer=reg)(x)
    model = Model(cb.input, x)
    #model.summary()
    #pr = tf.keras.metrics.AUC(name='PR', curve='PR')
    model.compile(optimizer=optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def get_data(data_dir, validation_size=0.05, test_size=0.2):
    diseased_dir = data_dir + 'diseased/'
    non_diseased_dir = data_dir + 'non_diseased/'

    start = time.time()
    print("\n[INFO] Gathering images and converting them to np arrays")
    diseased_images = np.asarray([cv2.imread(diseased_dir + im_path) for im_path in os.listdir(diseased_dir)])
    non_diseased_images = np.asarray([cv2.imread(non_diseased_dir + im_path) for im_path in os.listdir(non_diseased_dir) if 0.4 > np.random.random()])
    print("[INFO] Conversion took {} seconds".format(round(time.time() - start, 2)))

    #diseased_images = np.array([i for i in diseased_images])
    #non_diseased_images = np.array([i for i in non_diseased_images])
    np.random.shuffle(diseased_images)
    np.random.shuffle(non_diseased_images)

    d_test_size = int(len(diseased_images) * test_size)
    nd_test_size = int(len(non_diseased_images) * test_size)
    d_validation_size = int(len(diseased_images) * validation_size)
    nd_validation_size = int(len(diseased_images) * validation_size)

    test_diseased_images = diseased_images[-d_test_size:]
    test_non_diseased_images = non_diseased_images[-nd_test_size:]
    validation_diseased_images = diseased_images[-d_validation_size-d_test_size:-d_test_size]
    validation_non_diseased_images = non_diseased_images[-nd_validation_size-nd_test_size:-nd_test_size]
    diseased_images = diseased_images[:-d_validation_size-d_test_size]
    non_diseased_images = non_diseased_images[:-nd_validation_size-nd_test_size]

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

def experiment_1(data, batch_size, tissue_type, epochs=10, step_size=1, verbose=1):

    if verbose == 1:
        print('\n-------------------\n    Experiment 1\n-------------------')

    (diseased_images, non_diseased_images), (x_val, y_val), (x_test, y_test) = data

    # 5. Train model using TrainingSet and ValidationSet
    model = make_model(2)
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,  verbose=1, min_lr=1e-5)
    for i in range(0, epochs, step_size):
        nd = non_diseased_images [np.random.choice(len(non_diseased_images), len(diseased_images), replace=False)]
        train_data = np.concatenate([diseased_images, nd], axis=0)
        train_labels = np.concatenate([[0 for i in range(len(diseased_images))], [1 for i in range(len(nd))]], axis=0)
        x_train, y_train = shuffle(train_data, train_labels)
        h = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train)//batch_size, initial_epoch=i, epochs=i+step_size, validation_data=(x_val, y_val), callbacks=[es, reduce]).history
        history = update_history(history, h)
        tf.keras.backend.clear_session()
        gc.collect()

    # 6. Test model on TestingSet, report accuracy
    result = model.evaluate(x_test, y_test)
    cm = confusion_matrix(y_test, np.argmax(model.predict(x_test), axis=-1))
    print("confusion matrix")
    print(cm)
    print(accuracy_score(y_test, np.argmax(model.predict(x_test), axis=-1)))
    fpr, tpr, _ = roc_curve(y_test, model.predict(x_test))
    precision, recall, _ = precision_recall_curve(y_test, model.predict(x_test))
    roc_auc = roc_auc_score(y_test, model.predict(x_test))
    pr_auc = auc(recall, precision)
    plt.plot(np.arange(len(y_test))/len(y_test), np.arange(len(y_test))/len(y_test), linestyle='--')
    plt.plot(fpr, tpr, marker='.')
    plt.title("AUC: " + str(round(roc_auc, 4)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('./experiment4/' + tissue_type + '_Roc_Curve.png')
    plt.clf()
    #plt.show()
    plt.plot(np.arange(len(y_test))/len(y_test), np.zeros(len(y_test)), linestyle='--')
    plt.plot(recall, precision, marker='.')
    plt.title("AUC: " + str(round(pr_auc, 4)))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('./experiment4/' + tissue_type + '_Precision-Recall.png')
    plt.clf()
    #plt.show()

    if verbose == 1:
        print("Testing loss:", result[0])
        print("Testing accuracy:", result[1])
        print("confusion matrix:\n", cm)

    return result, model, history, cm

def experiment_2(data, batch_size, tissue_type, epochs=2, verbose=1):

    if verbose == 1:
        print('\n-------------------\n    Experiment 2\n-------------------')

    es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=1e-6)

    train_generator = datagen.flow_from_directory(data, target_size=(image_rez, image_rez),
                                                batch_size=batch_size, class_mode='sparse', subset='training')
    val_generator = datagen.flow_from_directory(data, target_size=(image_rez, image_rez),
                                                batch_size=batch_size, class_mode='sparse', subset='validation')
    model = make_model(train_generator.num_classes)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes), train_generator.classes)
    class_weights = dict(zip(range(len(class_weights)), class_weights))
    print(class_weights)
    history = model.fit(train_generator, steps_per_epoch=train_generator.samples // batch_size, epochs=epochs,
                        validation_data=val_generator, validation_steps=val_generator.samples // batch_size,
                        callbacks=[es, reduce], class_weight=class_weights)

    val_generator.reset()
    result = model.evaluate(val_generator, verbose=0)
    cm = confusion_matrix(val_generator.classes, np.argmax(model.predict(val_generator), axis=-1))
    rep = classification_report(val_generator.classes, np.argmax(model.predict(val_generator), axis=-1))
    print('result')
    print(result)
    print('cm')
    print(cm)
    print('sc')
    print(sc)
    print('report')
    print(rep)

    return result, model, history.history, cm


if __name__ == "__main__":

    #base_dir = '../data/tiled_datasets_full/'
    #base_dir = '../data/tiled_datasets/'
    base_dir = '../data/new_tiles/'
    tissue_type = sys.argv[1]
    #data_dir = base_dir + tissue_type + '_randomtiles/'
    data_dir = base_dir + tissue_type + '-tiles-new/binary/'
    batch_size = 32

    data = get_data(data_dir)

    result, model, history, cm = experiment_1(data, batch_size, tissue_type)
    #result, model, history, cm = experiment_2(sys.argv[2], batch_size, tissue_type)
    #experiment_3(data, output_size, batch_size)
    #'''
    with open("./new_tiles_experiment/{}_{}3.txt".format(tissue_type, sys.argv[2].split('/')[-2]), 'w') as f:
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
    #'''