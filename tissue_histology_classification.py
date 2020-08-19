from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


train_dir = 'data/randomtiles512/randomtiles512/train/'
test_dir = 'data/randomtiles512/randomtiles512/test/'


# extract convolutional base from Google's InceptionV3 object detection model
cb = InceptionResNetV2(weights='imagenet', 
                  include_top=False, 
                  input_shape=(512, 512, 3))
last_layer = layers.GlobalAveragePooling2D()(cb.output)
conv_base = Model(cb.input, last_layer)
#conv_base.summary()

def make_model():
    # creating trainable model to replace the head of InceptionV3
    model = models.Sequential()
    model.add(layers.Dropout(0.5, input_shape=(1536,)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(2, activation='softmax'))

    model.summary()
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# create data generator
datagen = ImageDataGenerator(rescale=1./255, rotation_range=90, width_shift_range=0.3, height_shift_range=0.3, shear_range=0.3, zoom_range=0.3, horizontal_flip=True, vertical_flip=True)
batch_size = 16

# returns the output of the last layer of the InceptionV3 convolutional neural network
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 1536), dtype=np.float32)
    labels = np.zeros(shape=(sample_count), dtype=np.float32)
    generator = datagen.flow_from_directory(
        directory,
        target_size=(512, 512),
        batch_size=batch_size,
        class_mode='sparse')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

# InceptionV3 output tensor
train_features, train_labels = extract_features(train_dir, 975)
train_features = np.reshape(train_features, (975, 1536))


if __name__ == "__main__":
    model1 = make_model()
    
    # fit the new model we have made to the training and validation data
    print('\nModel 1')
    h1 = model1.fit(train_features, train_labels, epochs=200, batch_size=16, validation_split=0.2)

    plt.figure(figsize=(10,8))
    plt.title('Batch Size: 1')
    plt.subplot(211)
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.plot(range(len(h1.history['accuracy'])), h1.history['accuracy'], label='training accuracy')
    plt.plot(range(len(h1.history['val_accuracy'])), h1.history['val_accuracy'], label='validation accuracy')
    plt.legend()
    plt.subplot(212)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.plot(range(len(h1.history['loss'])), h1.history['loss'], label='training loss')
    plt.plot(range(len(h1.history['val_loss'])), h1.history['val_loss'], label='validation loss')
    plt.legend()
    plt.show()

    print(np.average(h1.history['val_accuracy'][100:]))