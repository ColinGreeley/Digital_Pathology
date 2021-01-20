from tensorflow.keras.applications import MobileNetV2, InceptionResNetV2, NASNetLarge, NASNetMobile, ResNet152V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models, Model, regularizers, optimizers
from tensorflow.compat.v1.keras.layers import CuDNNLSTM, Masking
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

#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
lstm = layers.LSTM if gpu_devices == [] else CuDNNLSTM

rez = [224, 224, 224, 299, 331]
model = [MobileNetV2, ResNet152V2, NASNetMobile, InceptionResNetV2, NASNetLarge]
model_name = ['MobileNetV2', 'ResNet152V2', 'NASNetMobile', 'InceptionResNet', 'NASNetLarge']


class ConvLSTM:

    def __init__(self, i, l, batch_size):
        self.l = l
        self.batch_size = batch_size
        self.tile_size = rez[i]
        self.tile_increment = int(self.tile_size * 0.5)
        self.conv_base = model[i](weights='imagenet', include_top=False, input_shape=(self.tile_size, self.tile_size, 3), pooling='avg')
        self.datagen = ImageDataGenerator(rescale=1/255, horizontal_flip=True, vertical_flip=True, brightness_range=(0.7,1.3), rotation_range=20)
        self.make_model2()

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

        for k in range(10):
            for i in diseased_list[:]:
                start = time.time()
                image = cv2.imread(diseased_dir + i)
                tiles = []
                for (_, _, tile) in self.sliding_window(image, self.tile_size, self.tile_increment):
                    if not self.containsWhite(tile):
                        tiles.append(tile)
                diseased_features.append(self.conv_base.predict(self.datagen.flow(np.array(tiles))))
                print(i.split('.')[0], round(time.time()-start, 2))
        diseased_features = np.array(diseased_features)

        for i in non_diseased_list[:]:
            start = time.time()
            image = cv2.imread(non_diseased_dir + i)
            tiles = []
            for (_, _, tile) in self.sliding_window(image, self.tile_size, self.tile_increment):
                if not self.containsWhite(tile):
                    tiles.append(tile)
            non_diseased_features.append(self.conv_base.predict(self.datagen.flow(np.array(tiles))))
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
        #x = layers.Masking(mask_value=0.)(input_layer)
        x = layers.GaussianDropout(0.5)(input_layer)
        x = layers.Activation('relu')(x)
        x = layers.Bidirectional(lstm(256, return_sequences=True, kernel_regularizer=regularizers.l2(1e-3)))(x) #, kernel_regularizer=regularizers.l2(1e-4)
        x = layers.GaussianNoise(0.3)(x)
        x = layers.Bidirectional(lstm(128, return_sequences=True, kernel_regularizer=regularizers.l2(1e-3)))(x)
        x = layers.GaussianNoise(0.3)(x)
        x = layers.Bidirectional(lstm(64, kernel_regularizer=regularizers.l2(1e-3)))(x)
        x = layers.GaussianNoise(0.3)(x)
        x = layers.Dense(128)(x)
        x = layers.GaussianNoise(0.3)(x)
        x = layers.Activation('relu')(x)
        output_layer = layers.Dense(2, activation='softmax')(x)

        model = models.Model(input_layer, output_layer)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model
       
    def make_model2(self):
        reg = regularizers.l1_l2(1e-2, 1e-2)
        input_layer = layers.Input(shape=(self.l, self.conv_base.output.shape[-1]))
        x = layers.TimeDistributed(layers.Dense(512, kernel_regularizer=reg))(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.GaussianDropout(0.2)(x)
        x = layers.TimeDistributed(layers.Dense(256, kernel_regularizer=reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.GaussianDropout(0.2)(x)
        #x = layers.TimeDistributed(layers.Dense(128, kernel_regularizer=reg))(x)
        #x = layers.BatchNormalization()(x)
        #x = layers.LeakyReLU()(x)
        #x = layers.GaussianDropout(0.2)(x)


        #x = layers.Lambda(lambda x: K.sum(x, axis=1))(x)
        x = layers.GlobalMaxPooling1D()(x)
        #x = layers.Dense(256, kernel_regularizer=reg)(x)
        #x = layers.BatchNormalization()(x)
        #x = layers.LeakyReLU()(x)
        #x = layers.GaussianDropout(0.2)(x)
        x = layers.Dense(128, kernel_regularizer=reg)(x)
        x = layers.LeakyReLU()(x)
        x = layers.GaussianDropout(0.2)(x)
        output_layer = layers.Dense(2, activation='softmax')(x)

        model = models.Model(input_layer, output_layer)
        model.compile(optimizer=optimizers.Adam(epsilon=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        self.model.summary()

    def train(self, x, y): 
        x, v, y, w = train_test_split(x, y, test_size=0.2)
        reduce = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=30, min_lr=0.00001)
        #xx = np.repeat(x, 9, axis=0)
        #yy = np.repeat(y, 9, axis=0)
        #xa = np.append(x, xx, axis=0)
        #ya = np.append(y, yy, axis=0)
        #print(xa.shape, ya.shape)
        for i in range(1000):
            #print("Episode:", i+1)
            #for j in range(len(xa)):
            #    np.random.shuffle(xa[j])
            #xx = x + (np.random.normal(0, 0.1, size=x.shape) * x)
            #print('max x:', np.max(xx), ' mean x:', np.mean(xx), ' min x:', np.min(xx), '\n')
            history = self.model.fit(x, y, epochs=i+1, initial_epoch=i, verbose=1, batch_size=self.batch_size, validation_data=(v, w), callbacks=[])
            for j in range(len(x)):
                np.random.shuffle(x[j])
        return history


def create_dataset(crnn, model_name):
    df, ndf = crnn.extract_features(diseased_dir, non_diseased_dir)
    x_train, y_train = crnn.make_data(df, ndf)
    with open('x_{}.pkl'.format(model_name), 'wb') as f:
        pickle.dump(x_train, f)
    with open('y_{}.pkl'.format(model_name), 'wb') as f:
        pickle.dump(y_train, f)

def get_dataset(model_name):
    with open('x_{}.pkl'.format(model_name), 'rb') as f:
        x_train = pickle.load(f)
    with open('y_{}.pkl'.format(model_name), 'rb') as f:
        y_train = pickle.load(f)
    return (x_train, y_train)


if __name__ == "__main__":

    model_id = 3
    batch_size = 16
    diseased_dir = '../data/RNN_images/diseased/'
    non_diseased_dir = '../data/RNN_images/non_diseased/'
    
    x_train, y_train = get_dataset(model_name[model_id])
    CRNN = ConvLSTM(model_id, x_train.shape[1], batch_size)
    #df, ndf = CRNN.extract_features(diseased_dir, non_diseased_dir)
    #x_train, y_train = CRNN.make_data(df, ndf)
    print('\nx_shape:', x_train.shape, '\ny_shape:', y_train.shape)
    history = CRNN.train(x_train, y_train)
    print(history.history)
