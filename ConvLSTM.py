from tensorflow.keras.applications import MobileNet, InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models, Model, regularizers, optimizers
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import time
import pickle

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#gpu_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpu_devices[0], True)

rez = [244, 299]
model = [MobileNet, InceptionResNetV2]

class ConvLSTM:

    def __init__(self, i, batch_size):
        self.batch_size = batch_size    #B0  #B3  #B5
        self.tile_size = rez[i]         #224 #300 #456
        self.tile_increment = int(self.tile_size / 1)
        self.conv_base = model[i](weights='imagenet', include_top=False, input_shape=(self.tile_size, self.tile_size, 3), pooling='avg')
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
            start = time.time()
            image = cv2.imread(diseased_dir + i)
            tiles = []
            for (_, _, tile) in self.sliding_window(image, self.tile_size, self.tile_increment):
                if not self.containsWhite(tile):
                    tiles.append(tile)
            diseased_features.append(self.conv_base.predict(np.array(tiles) / 255.))
            print(i.split('.')[0], round(time.time()-start, 2))
        diseased_features = np.array(diseased_features)
        for i in non_diseased_list[:]:
            start = time.time()
            image = cv2.imread(non_diseased_dir + i)
            tiles = []
            for (_, _, tile) in self.sliding_window(image, self.tile_size, self.tile_increment):
                if not self.containsWhite(tile):
                    tiles.append(tile)
            non_diseased_features.append(self.conv_base.predict(np.array(tiles) / 255.))
            print(i.split('.')[0], round(time.time()-start, 2))
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
        y_train = np.array(y_train).astype('float32')

        print("x shape:", x_train.shape)
        print("y shape:", y_train.shape)
        return shuffle(x_train, y_train)
        
    def make_model(self):
        input_layer = layers.Input(shape=(None, self.conv_base.output.shape[-1]))
        x = layers.GaussianDropout(0.1)(input_layer)
        x = layers.Bidirectional(layers.LSTM(256, dropout=0.1, return_sequences=True, kernel_regularizer=regularizers.l1_l2(1e-3, 1e-3)))(x) #, kernel_regularizer=regularizers.l1_l2(1e-4, 1e-4)
        x = layers.GaussianNoise(0.2)(x)
        x = layers.Bidirectional(layers.LSTM(256, dropout=0.1, return_sequences=True, kernel_regularizer=regularizers.l1_l2(1e-3, 1e-3)))(x)
        x = layers.GaussianNoise(0.2)(x)
        x = layers.Bidirectional(layers.LSTM(256, dropout=0.1, kernel_regularizer=regularizers.l1_l2(1e-3, 1e-3)))(x)
        x = layers.GaussianNoise(0.2)(x)
        x = layers.Dense(128)(x)
        x = layers.GaussianNoise(0.2)(x)
        x = layers.Activation('relu')(x)
        output_layer = layers.Dense(2, activation='softmax')(x)

        model = models.Model(input_layer, output_layer)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        
    def train(self, x, y): 
        x, v, y, w = train_test_split(x_train, y_train, test_size=0.2)
        for i in range(1):
            print("Episode:", i+1)
            #xx = x + (np.random.normal(0, 0.1, size=x.shape) * x)
            #print('max x:', np.max(xx), ' mean x:', np.mean(xx), ' min x:', np.min(xx), '\n')
            self.model.fit(x, y, epochs=300, verbose=1, batch_size=self.batch_size, validation_data=(v, w))
            #for j in range(len(x)):
            #    np.random.shuffle(x[j])


def create_dataset(model_name, model_id, batch_size):
    CRNN = ConvLSTM(model_id, batch_size)
    df, ndf = CRNN.extract_features(diseased_dir, non_diseased_dir)
    x_train, y_train = CRNN.make_data(df, ndf)
    with open('x_{}.pkl'.format(model_name), 'wb') as f:
        pickle.dump(x_train, f)
    with open('y_{}.pkl'.format(model_name), 'wb') as f:
        pickle.dump(y_train, f)

def get_dataset(model_name, model_id):
    with open('x_{}.pkl'.format(model_name), 'rb') as f:
        x_train = pickle.load(f)
    with open('y_{}.pkl'.format(model_name), 'rb') as f:
        y_train = pickle.load(f)
    return (x_train, y_train)


if __name__ == "__main__":

    model_id = 1
    batch_size = 16
    diseased_dir = '../data/RNN_images/diseased/'
    non_diseased_dir = '../data/RNN_images/non_diseased/'
    
    x_train, y_train = get_dataset('InceptionResNet', model_id)
    CRNN = ConvLSTM(model_id, batch_size)
    #df, ndf = CRNN.extract_features(diseased_dir, non_diseased_dir)
    #x_train, y_train = CRNN.make_data(df, ndf)
    print('\nx_shape:', x_train.shape, '\ny_shape:', y_train.shape)
    CRNN.train(x_train, y_train)
