import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras import layers, models, Model
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import os


def make_model():
    conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    for layer in conv_base.layers:
        layer.trainable = False
    pooling = layers.GlobalMaxPooling2D()(conv_base.output)
    net = layers.Dropout(0.5)(pooling)
    output = layers.Dense(2, activation='softmax')(net)

    model = Model(conv_base.input, output)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def get_data():
    data_dir = './data/randomtiles512/randomtiles512/'
    diseased_dir = data_dir + 'diseased/'
    non_diseased_dir = data_dir + 'non_diseased/'
    
    diseased_images = [Image.open(diseased_dir + im_path).resize((512,512)) for im_path in os.listdir(diseased_dir)]
    non_diseased_images = [Image.open(non_diseased_dir + im_path).resize((512,512)) for im_path in os.listdir(non_diseased_dir)]
    # normalize pixel values with 1/255
    diseased_images = np.array([np.asarray(i) / 255. for i in diseased_images])
    non_diseased_images = np.array([np.asarray(i) / 255. for i in non_diseased_images])

    print("\nDiseased image folder shape:", diseased_images.shape)
    print("Non_diseased image folder shape:", non_diseased_images.shape)

    data = np.append(diseased_images, non_diseased_images, axis=0)
    # diseased <= [1, 0], non_diseased <= [0, 1] for categorical labels
    labels = np.append([[0] for i in range(len(diseased_images))], [[1] for i in range(len(non_diseased_images))], axis=0)
    
    return data, labels
    


def experiment_1(X, y, model):
    # 1. Let LearningSet = 0.8*N images selected at random from X
    # 2. Let TestingSet = X – LearningSet
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    # 3. Let ValidationSet = 0.2*|LearningSet| images selected at random from LearningSet
    # 4. TrainingSet = LearningSet – ValidationSet
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    
    # 5. Train model using TrainingSet and ValidationSet
    model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val))

    # 6. Test model on TestingSet, report accuracy
    model.evaluate(x_test, y_test)


def experiment_2():
    # Repeat Experiment 1 multiple times and report mean accuracy and variance
    pass

def experiment_3():
    # 1. Randomly partition X in to v TestingSets of N/v images each
    # 2. For i = 1 to v
    #        Let LearningSet = X – TestingSet_i
    #        Let ValidationSet = 0.2*|LearningSet| images selected at random from LearningSet
    #        TrainingSet = LearningSet – ValidationSet
    #        Train model using TrainingSet and ValidationSet
    #        Test model on TestingSet_i, retain accuracy_i
    # 3. Report mean and variance over accuracy_i
    pass
    



if __name__ == "__main__":
    X, y = get_data()
    print(X.shape, y.shape)
    model = make_model()

    experiment_1(X, y, model)