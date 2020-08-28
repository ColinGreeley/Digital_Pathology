import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras import layers, models, Model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


def make_model():
    conv_base = InceptionResNetV2(weights='imagenet', 
                                include_top=False, 
                                input_shape=(512, 512, 3))
    for layer in conv_base.layers:
        layer.trainable = False
    net = layers.GlobalAveragePooling2D()(conv_base.output)
    net = layers.Dense(128, activation='relu')(net)
    net = layers.Dropout(0.3)(net)
    output = layers.Dense(2, activation='softmax')(net)
    model = Model(conv_base.input, output)

    #model.summary()
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def make_data_generator():
    generator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20, 
                zoom_range=0.1,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=True)
    return generator

def get_data(data_dir):
    diseased_dir = data_dir + 'diseased/'
    non_diseased_dir = data_dir + 'non_diseased/'
    
    diseased_images = [Image.open(diseased_dir + im_path).resize((512,512)) for im_path in os.listdir(diseased_dir)]
    non_diseased_images = [Image.open(non_diseased_dir + im_path).resize((512,512)) for im_path in os.listdir(non_diseased_dir)]
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


def experiment_1(X, y, batch_size, verbose=1):

    if verbose == 1:
        print('\n-------------------\n    Experiment 1\n-------------------')
    
    X, y = shuffle(X, y)
    train_generator = make_data_generator()

    # 1. Let LearningSet = 0.8*N images selected at random from X
    # 2. Let TestingSet = X – LearningSet
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    
    # 3. Let ValidationSet = 0.2*|LearningSet| images selected at random from LearningSet
    # 4. TrainingSet = LearningSet – ValidationSet
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    
    # 5. Train model using TrainingSet and ValidationSet
    model = make_model()
    model.fit(train_generator.flow(x_train, y_train, batch_size=batch_size), 
                steps_per_epoch=len(x_train)/batch_size,
                validation_data=(x_val, y_val), epochs=300, verbose=1)
    
    # 6. Test model on TestingSet, report accuracy
    result = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    if verbose == 1:
        print("Testing loss:", result[0])
        print("Testing accuracy:", result[1])

    return result[1]


def experiment_2(X, y, batch_size, experiment_count=10):
    
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

    
def experiment_3(X, y, batch_size, v=10): 

    print('\n-------------------\n    Experiment 3\n-------------------')

    train_generator = make_data_generator()

    # 1. Randomly partition X in to v TestingSets of N/v images each
    x_partitions = np.array_split(X, v)
    y_partitions = np.array_split(y, v)

    results = []

    # 2. For i = 1 to v
    for i in range(1, v):
        print("Fold " + str(i) + '/' + str(v-1))

    #   Let LearningSet = X – TestingSet_i
        x_train = np.concatenate([x_partitions[j] for j in range(v) if j != i])
        y_train = np.concatenate([y_partitions[j] for j in range(v) if j != i])
        x_test = x_partitions[i]
        y_test = y_partitions[i]

    #   Let ValidationSet = 0.2*|LearningSet| images selected at random from LearningSet
    #   TrainingSet = LearningSet – ValidationSet
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    #   Train model using TrainingSet and ValidationSet
        model = make_model()
        model.fit(train_generator.flow(x_train, y_train, batch_size=batch_size), 
                    steps_per_epoch=len(x_train)/batch_size, 
                    validation_data=(x_val, y_val), epochs=300, verbose=0)

    #   Test model on TestingSet_i, retain accuracy_i
        result = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
        results.append(result[1])

    # 3. Report mean and variance over accuracy_i
    mean_accuracy = np.mean(results)
    variance = np.var(results)
    print("Mean test accuracy:", mean_accuracy)
    print("Variance:", variance)

    

if __name__ == "__main__":

    data_dir = '../data/randomtiles/randomtiles/'
    batch_size = 32
    
    X, y = get_data(data_dir)
    
    experiment_1(X, y, batch_size)
    experiment_2(X, y, batch_size)
    experiment_3(X, y, batch_size)