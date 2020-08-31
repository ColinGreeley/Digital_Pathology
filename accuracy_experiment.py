import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras import layers, models, Model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image
import os


image_rez = 256 #512

def make_feature_extractor():
    conv_base = InceptionResNetV2(weights='imagenet', 
                                include_top=False, 
                                input_shape=(image_rez, image_rez, 3))
    output = layers.GlobalAveragePooling2D()(conv_base.output)
    return Model(conv_base.input, output), output.shape[-1]

def make_model(input_size):
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(input_size,)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(2, activation='softmax'))

    #model.summary()
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def get_data(data_dir):
    diseased_dir = data_dir + 'diseased/'
    non_diseased_dir = data_dir + 'non_diseased/'
    
    diseased_images = [Image.open(diseased_dir + im_path).resize((image_rez,image_rez)) for im_path in os.listdir(diseased_dir)]
    non_diseased_images = [Image.open(non_diseased_dir + im_path).resize((image_rez,image_rez)) for im_path in os.listdir(non_diseased_dir)]
    # rescale RGB values from 0-255 to 0-1
    diseased_images = np.array([np.asarray(i) / 255. for i in diseased_images])
    non_diseased_images = np.array([np.asarray(i) / 255. for i in non_diseased_images])

    print("\nDiseased image folder shape:", diseased_images.shape)
    print("Non_diseased image folder shape:", non_diseased_images.shape)
    print("Total images:", len(diseased_images) + len(non_diseased_images))

    data = np.append(diseased_images, non_diseased_images, axis=0)
    labels = np.append([0 for i in range(len(diseased_images))], [1 for i in range(len(non_diseased_images))], axis=0)
    
    # shuffle the data and labels together
    data, labels = shuffle(data, labels)
    return data, labels

def extract_features(X, y, feature_extractor):
    features = feature_extractor.predict(X)
    return features, y


def experiment_1(X, y, input_size, batch_size, verbose=1):

    if verbose == 1:
        print('\n-------------------\n    Experiment 1\n-------------------')

    X, y = shuffle(X, y)

    # 1. Let LearningSet = 0.8*N images selected at random from X
    # 2. Let TestingSet = X – LearningSet
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    
    # 3. Let ValidationSet = 0.2*|LearningSet| images selected at random from LearningSet
    # 4. TrainingSet = LearningSet – ValidationSet
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    
    # 5. Train model using TrainingSet and ValidationSet
    model = make_model(input_size)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=300, validation_data=(x_val, y_val), verbose=0)
    
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

    data_dir = '../data/randomtiles256/randomtiles256/'
    batch_size = 32
    
    feature_extractor, output_size = make_feature_extractor()
    X, y = get_data(data_dir)
    X, y = extract_features(X, y, feature_extractor)
    
    experiment_1(X, y, output_size, batch_size)
    experiment_2(X, y, output_size, batch_size)
    experiment_3(X, y, output_size, batch_size)