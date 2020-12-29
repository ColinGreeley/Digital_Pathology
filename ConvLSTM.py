from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models, Model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class ConvLSTM:

    def __init__(self):
        self.tile_size = 256
        self.tile_increment = self.tile_size // 1
        self.conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(self.tile_size, self.tile_size, 3), pooling='avg')
        self.make_model()

    def sliding_window(self, image, window_size, step):
        for y in range(0, image.shape[0] - window_size, step):
            for x in range(0, image.shape[1] - window_size, step):
                yield (x, y, image[y:y + window_size, x:x + window_size])

    def containsWhite(self, image):
        avg = np.mean(image)
        return avg == 255

    def extract_features(self, diseased_features, non_diseased_features):
        diseased_features = []
        non_diseased_features = []
        diseased_list = os.listdir(diseased_dir)
        non_diseased_list = os.listdir(non_diseased_dir)
        for i in diseased_list[:]:
            print(i)
            image = cv2.imread(diseased_dir + i)
            tiles = []
            for (_, _, tile) in self.sliding_window(image, self.tile_size, self.tile_increment):
                if not self.containsWhite(tile):
                    tiles.append(tile)
            diseased_features.append(self.conv_base.predict(np.array(tiles) / 255.))
        diseased_features = np.array(diseased_features)
        for i in non_diseased_list[:]:
            print(i)
            image = cv2.imread(non_diseased_dir + i)
            tiles = []
            for (_, _, tile) in self.sliding_window(image, self.tile_size, self.tile_increment):
                if not self.containsWhite(tile):
                    tiles.append(tile)
            non_diseased_features.append(self.conv_base.predict(np.array(tiles) / 255.))
        non_diseased_features = np.array(non_diseased_features)
        return (diseased_features, non_diseased_features)

    def make_data(self, diseased_features, non_diseased_features):
        x_train = []
        y_train = []
        for i in diseased_features:
            x_train.append(i)
            y_train.append(0)
        for i in non_diseased_features:
            x_train.append(i)
            y_train.append(1)

        x_train = np.array(x_train)
        x_train = pad_sequences(x_train, dtype='float32')
        y_train = np.array(y_train)

        print("x shape:", x_train.shape)
        print("y shape:", y_train.shape)
        return (x_train, y_train)

    def make_model(self):
        input_layer = layers.Input(shape=(None, self.conv_base.output.shape[-1]))
        x = layers.LSTM(256, dropout=0.2, return_sequences=True)(input_layer)
        #x = layers.LSTM(128, dropout=0.2, return_sequences=True)(x)
        x = layers.LSTM(64, dropout=0.2)(x)
        output_layer = layers.Dense(2, activation='softmax')(x)

        model = models.Model(input_layer, output_layer)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def train(self, x, y, batch_size):
        self.model.fit(x_train, y_train, epochs=50, batch_size=batch_size, shuffle=True)



if __name__ == "__main__":

    CRNN = ConvLSTM()
    diseased_dir = '../data/RNN_images/diseased/'
    non_diseased_dir = '../data/RNN_images/non_diseased/'

    df, ndf = CRNN.extract_features(diseased_dir, non_diseased_dir)
    x_train, y_train = CRNN.make_data(df, ndf)
    CRNN.train(x_train, y_train, batch_size=8)
