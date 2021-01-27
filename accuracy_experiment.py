import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2, EfficientNetB6
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Model, optimizers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image
import cv2
import os

physical_devices = tf.config.experimental.list_physical_devices('GPU')
#if physical_devices != []:
#config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

image_rez = 256
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, brightness_range=(0.5,1.5), rotation_range=40, channel_shift_range=100)

def make_feature_extractor():
    conv_base = EfficientNetB2(weights='imagenet', 
                                include_top=False, 
                                input_shape=(image_rez, image_rez, 3))
    output = layers.GlobalAveragePooling2D()(conv_base.output)
    return Model(conv_base.input, output), output.shape[-1]

def make_model(input_size):
    conv_base = EfficientNetB2(weights='imagenet', 
                                include_top=False, 
                                drop_connect_rate=0.4,
                                pooling='avg',
                                input_shape=(image_rez, image_rez, 3))
    #conv_base.trainable = False
    i = 0
    while True:
        conv_base.layers[i].trainable = False
        if 'block3' in conv_base.layers[i].name:
            break
        i += 1
    
    x = conv_base.output
    #x = layers.BatchNormalization()(x)
    #x = layers.GaussianNoise(0.5)(x)
    #x = layers.Dense(256, activation='relu', kernel_regularizer='l1_l2')(x)
    x = layers.GaussianNoise(1)(x)
    x = layers.Dense(2, activation='softmax', kernel_regularizer='l1_l2')(x)

    model = Model(conv_base.input, x)
    model.summary()
    model.compile(optimizer=optimizers.Adam(0.0005, epsilon=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def get_data(data_dir):
    diseased_dir = data_dir + 'diseased/'
    non_diseased_dir = data_dir + 'non_diseased/'
    
    diseased_images = [cv2.imread(diseased_dir + im_path) for im_path in os.listdir(diseased_dir)]
    non_diseased_images = [cv2.imread(non_diseased_dir + im_path) for im_path in os.listdir(non_diseased_dir)]
    
    diseased_images = np.array([cv2.resize(i, (image_rez,image_rez)) / 1. for i in diseased_images])
    non_diseased_images = np.array([cv2.resize(i, (image_rez,image_rez)) / 1. for i in non_diseased_images]) #[np.random.choice(len(non_diseased_images), int(len(non_diseased_images)/4), replace=False)]

    print("\nDiseased image folder shape:", diseased_images.shape)
    print("Non_diseased image folder shape:", non_diseased_images.shape)
    print("Total images:", len(diseased_images) + len(non_diseased_images))

    data = np.concatenate([diseased_images, non_diseased_images], axis=0)
    labels = np.concatenate([[0 for i in range(len(diseased_images))], [1 for i in range(len(non_diseased_images))]], axis=0)
    
    # shuffle the data and labels together
    data, labels = shuffle(data, labels)
    return data, labels

def extract_features(X, y, feature_extractor):
    features = feature_extractor.predict(X, batch_size=8)
    return features, y


def experiment_1(X, y, input_size, batch_size, verbose=1):

    if verbose == 1:
        print('\n-------------------\n    Experiment 1\n-------------------')

    # 1. Let LearningSet = 0.8*N images selected at random from X
    # 2. Let TestingSet = X – LearningSet
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    
    # 3. Let ValidationSet = 0.2*|LearningSet| images selected at random from LearningSet
    # 4. TrainingSet = LearningSet – ValidationSet
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    
    # 5. Train model using TrainingSet and ValidationSet
    model = make_model(input_size)
    model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train)//batch_size, epochs=300, validation_data=(x_val, y_val), class_weight={0:1, 1:0.25})
    
    # 6. Test model on TestingSet, report accuracy
    result = model.evaluate(x_test, y_test, verbose=0)
    cm = confusion_matrix(y_test, np.argmax(model.predict(x_test), axis=-1))
    if verbose == 1:
        print("Testing loss:", result[0])
        print("Testing accuracy:", result[1])
        print("confusion matrix:\n", cm)

    return result[1]


def experiment_2(X, y, input_size, batch_size, experiment_count=10):
    
    print('\n-------------------\n    Experiment 2\n-------------------')

    # Repeat Experiment 1 multiple times and report mean accuracy and variance
    results = []

    for i in range(experiment_count):
        print("Run " + str(i+1) + '/' + str(experiment_count))
        results.append(experiment_1(X, y, input_size, batch_size, verbose=0))
    
    mean_accuracy = np.mean(results)
    variance = np.var(results)
    print("Mean test accuracy:", mean_accuracy)
    print("Variance:", variance)


def experiment_3(X, y, input_size, batch_size, v=10): 

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

    #data_dir = '../data/randomtiles512/randomtiles512/'
    data_dir = '../data/randomtiles256/randomtiles256/'
    #data_dir = '../data/randomtiles-testes/randomtiles-testes/'
    batch_size = 16
    
    feature_extractor, output_size = make_feature_extractor()
    X, y = get_data(data_dir)
    #X, y = extract_features(X, y, feature_extractor)
    
    experiment_1(X, y, output_size, batch_size)
    experiment_2(X, y, output_size, batch_size)
    experiment_3(X, y, output_size, batch_size)