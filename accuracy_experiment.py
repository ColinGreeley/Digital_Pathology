"""
Colin Greeley
Accuracy Experiment
Used to train tile classifying models and to record the training and testing metrics

Usage: python3 accuracy_experiment.py <tissue_type> <model_size>
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

def make_feature_extractor():
    conv_base = EfficientNetB2(weights='imagenet',
                                include_top=False,
                                input_shape=(image_rez, image_rez, 3))
    output = layers.GlobalAveragePooling2D()(conv_base.output)
    return Model(conv_base.input, output), output.shape[-1]

def freeze_model(m, block):
    #m.trainable = True
    i = 0
    while True:
        if 'block{}'.format(block) in m.layers[i].name:
            break
        m.layers[i].trainable = False
        i += 1

def custom_loss(y_true, y_pred):  # crossentropy with increased loss for false positives
    #loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred) * fp
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    yy = (y_true.numpy() == 1) * (y_pred.numpy() < 0.5) * 100
    return loss * yy

def make_model(ensemble_size=3):
    input_layer = layers.Input(shape=(256, 256, 3))
    ensemble = []
    for i in range(ensemble_size):
        ensemble.append(EfficientNetB2(weights='imagenet', include_top=False, drop_connect_rate=0.4, pooling='avg', input_tensor=input_layer))
        for layer in ensemble[i].layers:
            layer._name = str(layer._name) + '_' + str(i)
        freeze_model(ensemble[i], 4)

    ensemble_outputs = [ensemble[i].output for i in range(ensemble_size)]
    x = layers.Concatenate()(ensemble_outputs)
    x = layers.GaussianNoise(1.0)(x)
    x = layers.Dense(2, activation='softmax', kernel_regularizer='l1_l2')(x)

    model = Model(input_layer, x)
    #model.summary()
    model.compile(optimizer=optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def get_data(data_dir):
    diseased_dir = data_dir + 'diseased/'
    non_diseased_dir = data_dir + 'non_diseased/'

    start = time.time()
    print("\n[INFO] Gathering images and converting them to np arrays")
    diseased_images = np.asarray([cv2.imread(diseased_dir + im_path) for im_path in os.listdir(diseased_dir)])
    non_diseased_images = np.asarray([cv2.imread(non_diseased_dir + im_path) for im_path in os.listdir(non_diseased_dir) if 2.4 > np.random.random()])
    print("[INFO] Conversion took {} seconds".format(round(time.time() - start, 2)))

    #diseased_images = np.array([i for i in diseased_images])
    #non_diseased_images = np.array([i for i in non_diseased_images])
    np.random.shuffle(diseased_images)
    np.random.shuffle(non_diseased_images)

    test_size = int(len(diseased_images) * 0.2)
    validation_size = int(len(diseased_images) * 0.2)

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

def experiment_1(data, batch_size, epochs=300, step_size=1, verbose=1):

    if verbose == 1:
        print('\n-------------------\n    Experiment 1\n-------------------')

    (diseased_images, non_diseased_images), (x_val, y_val), (x_test, y_test) = data

    # 5. Train model using TrainingSet and ValidationSet
    model = make_model()
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)
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


def experiment_2(data, input_size, batch_size, experiment_count=5):

    print('\n-------------------\n    Experiment 2\n-------------------')

    # Repeat Experiment 1 multiple times and report mean accuracy and variance
    results = []

    for i in range(experiment_count):
        print("Run " + str(i+1) + '/' + str(experiment_count))
        results.append(experiment_1(data, input_size, batch_size, verbose=0))

    mean_accuracy = np.mean(results)
    variance = np.var(results)
    print("Mean test accuracy:", mean_accuracy)
    print("Variance:", variance)


def experiment_3(X, y, input_size, batch_size, v=5):

    print('\n-------------------\n    Experiment 3\n-------------------')

    X, y = shuffle(X, y)
    results = []

    # 1. Randomly partition X in to v TestingSets of N/v images each
    x_partitions = np.array_split(X, v)
    y_partitions = np.array_split(y, v)

    # 2. For i = 1 to v
    for i in range(0, v):
        print("Fold " + str(i+1) + '/' + str(v))

    #   Let LearningSet = X – TestingSet_i
        x_train = np.concatenate([x_partitions[j] for j in range(v) if j != i])
        y_train = np.concatenate([y_partitions[j] for j in range(v) if j != i])
        x_test = x_partitions[i]
        y_test = y_partitions[i]

    #   Let ValidationSet = 0.2*|LearningSet| images selected at random from LearningSet
    #   TrainingSet = LearningSet – ValidationSet
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    #   Train model using TrainingSet and ValidationSet
        model = make_model(input_size)
        model.fit(x_train, y_train, batch_size=batch_size, epochs=300, validation_data=(x_val, y_val), verbose=0)

    #   Test model on TestingSet_i, retain accuracy_i
        results.append(model.evaluate(x_test, y_test, verbose=0)[1])

    # 3. Report mean and variance over accuracy_i
    mean_accuracy = np.mean(results)
    variance = np.var(results)
    print("Mean test accuracy:", mean_accuracy)
    print("Variance:", variance)



if __name__ == "__main__":

    base_dir = './tiled_datasets_large/'
    #data_dir = base_dir + 'testes' + '_randomtiles' + '1024' + '/'
    if len(sys.argv) < 3:
        tissue_type = ''
    else:
        tissue_type = sys.argv[2]
    data_dir = base_dir + sys.argv[1] + '_randomtiles' + tissue_type + '/'
    batch_size = 8

    feature_extractor, output_size = make_feature_extractor()
    data = get_data(data_dir)

    result, model, history, cm = experiment_1(data, batch_size)
    #experiment_2(data, output_size, batch_size)
    #experiment_3(data, output_size, batch_size)

    model.save_weights("./experiment3/{}{}_weights.h5".format(sys.argv[1], tissue_type))
    with open("./experiment3/{}{}.txt".format(sys.argv[1], tissue_type), 'w') as f:
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