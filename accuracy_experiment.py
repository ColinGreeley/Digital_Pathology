import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras import layers, models, Model
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import os


def make_feature_extractor():
    conv_base = InceptionResNetV2(weights='imagenet', 
                                include_top=False, 
                                input_shape=(512, 512, 3))
    output = layers.GlobalAveragePooling2D()(conv_base.output)
    return Model(conv_base.input, output), output.shape[-1]

def make_model(input_size):
    model = models.Sequential()
    model.add(layers.Dropout(0.3, input_shape=(input_size,)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(2, activation='softmax'))

    #model.summary()
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def make_data_generator(test=False):
    if test:
        generator = tf.keras.preprocessing.image.ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=40, 
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    horizontal_flip=True,
                    vertical_flip=True,
                    zoom_range=0.2)
    else:
        generator = tf.keras.preprocessing.image.ImageDataGenerator(
                    rescale=1./255)
    return generator

def get_data(data_dir):
    diseased_dir = data_dir + 'diseased/'
    non_diseased_dir = data_dir + 'non_diseased/'
    
    diseased_images = [Image.open(diseased_dir + im_path).resize((512,512)) for im_path in os.listdir(diseased_dir)]
    non_diseased_images = [Image.open(non_diseased_dir + im_path).resize((512,512)) for im_path in os.listdir(non_diseased_dir)]
    # normalize pixel values with 1/255
    diseased_images = np.array([np.asarray(i) / 1. for i in diseased_images])
    non_diseased_images = np.array([np.asarray(i) / 1. for i in non_diseased_images])

    print("\nDiseased image folder shape:", diseased_images.shape)
    print("Non_diseased image folder shape:", non_diseased_images.shape)
    print("Total images:", len(diseased_images) + len(non_diseased_images))

    data = np.append(diseased_images, non_diseased_images, axis=0)
    # diseased <= [1, 0], non_diseased <= [0, 1] for categorical labels
    labels = np.append([0 for i in range(len(diseased_images))], [1 for i in range(len(non_diseased_images))], axis=0)
    return data, labels

def extract_features(data, sample_count, input_size, datagen, feature_extractor, batch_size):
    features = np.zeros(shape=(sample_count, input_size), dtype=np.float32)
    labels = np.zeros(shape=(sample_count), dtype=np.float32)
    generator = datagen.flow(data[0], data[1], batch_size=batch_size)
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = feature_extractor.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return np.reshape(features, (sample_count, input_size)), labels


def experiment_1(X, y, input_size, feature_extractor, batch_size):

    print('\n-------------------\n    Experiment 1\n-------------------')
    
    train_generator = make_data_generator()
    test_generator = make_data_generator(test=True)

    # 1. Let LearningSet = 0.8*N images selected at random from X
    # 2. Let TestingSet = X – LearningSet
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    
    # 3. Let ValidationSet = 0.2*|LearningSet| images selected at random from LearningSet
    # 4. TrainingSet = LearningSet – ValidationSet
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    
    print("***Extracting training features***")
    print("Train data samples:", x_train.shape[0])
    x_train, y_train = extract_features(data=(x_train, y_train),
                                        sample_count=len(x_train),
                                        input_size=input_size,
                                        datagen=train_generator,
                                        feature_extractor=feature_extractor,
                                        batch_size=batch_size)
    print("***Extracting validation features***")
    print("Validation data samples:", x_val.shape[0])
    x_val, y_val = extract_features(data=(x_val, y_val),
                                        sample_count=len(x_val),
                                        input_size=input_size,
                                        datagen=test_generator,
                                        feature_extractor=feature_extractor,
                                        batch_size=batch_size)
    print("***Extracting testing features***")
    print("Test data samples:", x_test.shape[0])
    x_test, y_test = extract_features(data=(x_test, y_test),
                                        sample_count=len(x_test),
                                        input_size=input_size,
                                        datagen=test_generator,
                                        feature_extractor=feature_extractor,
                                        batch_size=batch_size)

    # 5. Train model using TrainingSet and ValidationSet
    model = make_model(input_size)
    print("\n****Training****")
    model.fit(x_train, y_train, batch_size=batch_size, epochs=100, 
                validation_data=(x_val, y_val), verbose=0)
    
    # 6. Test model on TestingSet, report accuracy
    result = model.evaluate(x_test, y_test, verbose=0)
    print("Testing loss:", result[0])
    print("Testing accuracy:", result[1])

    return ((x_train, y_train), (x_val, y_val), (x_test, y_test))


def experiment_2(data, batch_size, input_size, experiment_count=10):
    
    print('\n-------------------\n    Experiment 2\n-------------------')

    # Repeat Experiment 1 multiple times and report mean accuracy and variance
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    results = list()

    for i in range(experiment_count):
        print("Run " + str(i) + '/' + str(experiment_count))
        model = make_model(input_size)
        model.fit(x_train, y_train, batch_size=batch_size, epochs=100, 
                    validation_data=(x_val, y_val), verbose=0)
        results.append(model.evaluate(x_test, y_test, verbose=0)[1])
    
    mean_accuracy = np.mean(results)
    variance = np.var(results)
    print("Mean accuracy:", mean_accuracy)
    print("Variance:", variance)

    
def experiment_3():

    print('\n-------------------\n    Experiment 3\n-------------------')
    # 1. Randomly partition X in to v TestingSets of N/v images each
    # 2. For i = 1 to v
    #        Let LearningSet = X – TestingSet_i
    #        Let ValidationSet = 0.2*|LearningSet| images selected at random from LearningSet
    #        TrainingSet = LearningSet – ValidationSet
    #        Train model using TrainingSet and ValidationSet
    #        Test model on TestingSet_i, retain accuracy_i
    # 3. Report mean and variance over accuracy_i
    
    


if __name__ == "__main__":

    data_dir = '../data/randomtiles512/randomtiles512/'
    batch_size = 32
    picture_count = (len(os.listdir(data_dir + 'diseased/')) + 
                    len(os.listdir(data_dir + 'non_diseased/')))
    
    feature_extractor, output_size = make_feature_extractor()
    X, y = get_data(data_dir)
    model = make_model(output_size)
    
    extracted_data = experiment_1(X, y, output_size, feature_extractor, batch_size)
    experiment_2(extracted_data, batch_size, output_size)