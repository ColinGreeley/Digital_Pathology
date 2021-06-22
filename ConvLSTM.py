from tensorflow.keras.applications import VGG16, InceptionV3, EfficientNetB2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models, Model, regularizers, optimizers
from tensorflow.compat.v1.keras.layers import CuDNNLSTM, Masking
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split, KFold
import tensorflow.keras.backend as K
from sklearn import svm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import cv2
from PIL import Image
import time
import pickle

#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices != []:
    config = tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    lstm = CuDNNLSTM
else:
    lstm = layers.LSTM

#rez = [224, 224, 224, 299, 331]
rez = [224, 256]
mOdels = [VGG16, EfficientNetB2]
#model_name = ['MobileNetV2', 'ResNet152V2', 'NASNetMobile', 'InceptionResNet', 'NASNetLarge']
model_names = ['VGG16', 'EfficientNetB2']


class ConvLSTM:

    def __init__(self, i, batch_size):
        self.batch_size = batch_size
        self.tile_size = rez[i]
        self.model_name = model_names[i]
        self.tile_increment = int(self.tile_size * 1)
        self.conv_base = mOdels[i](weights='imagenet', include_top=False, input_shape=(self.tile_size, self.tile_size, 3), pooling='avg')
        #self.conv_base = Model(loaded_model.input, loaded_model.layers[-3].output)

    def sliding_window(self, image, window_size, step):
        for y in range(0, image.shape[0] - window_size, step):
            for x in range(0, image.shape[1] - window_size, step):
                yield (x, y, image[y:y + window_size, x:x + window_size])

    def containsWhite(self, image):
        avg = np.mean(image)
        return avg == 255

    def get_tiles(self, diseased_features, non_diseased_features):
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
            diff = 255.0 if self.model_name == 'VGG16' else 1.0
            diseased_features.append(np.asarray(tiles).astype('float32') / diff)
            print(i.split('.')[0], round(time.time()-start, 2))
        diseased_features = np.asarray(diseased_features)

        for i in non_diseased_list[:]:
            start = time.time()
            image = cv2.imread(non_diseased_dir + i)
            tiles = []
            for (_, _, tile) in self.sliding_window(image, self.tile_size, self.tile_increment):
                if not self.containsWhite(tile):
                    tiles.append(tile)
            diff = 255.0 if self.model_name == 'VGG16' else 1.0
            non_diseased_features.append(np.asarray(tiles).astype('float32') / diff)
            print(i.split('.')[0], round(time.time()-start, 2))
        non_diseased_features = np.asarray(non_diseased_features)

        return (diseased_features, non_diseased_features)


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
            diseased_features.append(self.conv_base.predict(np.array(tiles)))
            print(i.split('.')[0], round(time.time()-start, 2))
        diseased_features = np.array(diseased_features)

        for i in non_diseased_list[:]:
            start = time.time()
            image = cv2.imread(non_diseased_dir + i)
            tiles = []
            for (_, _, tile) in self.sliding_window(image, self.tile_size, self.tile_increment):
                if not self.containsWhite(tile):
                    tiles.append(tile)
            non_diseased_features.append(self.conv_base.predict(np.array(tiles)))
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

        x_train = np.asarray(x_train)
        x_train = pad_sequences(x_train, dtype='float32')
        y_train = np.asarray(y_train).astype('float32')

        print("x shape:", x_train.shape)
        print("y shape:", y_train.shape)
        return shuffle(x_train, y_train)

    def make_conv_base(self, weights):
        cb = EfficientNetB2(weights='imagenet',
                                    include_top=False,
                                    drop_connect_rate=0.4,
                                    pooling='avg',
                                    input_shape=(self.tile_size, self.tile_size, 3))
        x = cb.output
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(2, activation='softmax', kernel_regularizer='l1_l2')(x)

        model = Model(cb.input, x)
        model.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.load_weights(weights)
        self.conv_base = Model(model.input, model.layers[-3].output)

    def make_model(self, l):
        reg = regularizers.l1_l2(0.0001, 0.0001)
        input_layer = layers.Input(shape=(l[1], l[2]))
        #x = lstm(128, return_sequences=True)(input_layer)
        #x = lstm(128, return_sequences=True)(x)
        x = lstm(128, return_sequences=False, kernel_regularizer=reg)(input_layer)
        #x = lstm(64, return_sequences=False, kernel_regularizer=reg)(x)
        #x = layers.Dense(32, activation='relu')(x)
        #x = layers.Dropout(0.2)(x)
        x = layers.GaussianNoise(0.5)(x)
        output_layer = layers.Dense(2, activation='softmax', kernel_regularizer='l1_l2')(x)

        model = models.Model(input_layer, output_layer)
        model.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def make_model2(self, l):
        reg = regularizers.l1_l2(1e-3, 1e-3)
        input_layer = layers.Input(shape=(l[1], l[2]))
        x = layers.GlobalMaxPooling1D()(input_layer)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        output_layer = layers.Dense(2, activation='softmax', kernel_regularizer='l1_l2')(x)

        model = models.Model(input_layer, output_layer)
        model.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def make_model3(self, l):
        reg = regularizers.l1_l2(1e-2, 1e-2)
        input_layer = layers.Input(shape=(l[1], l[2]))
        x = layers.Flatten(input_layer)
        x = layers.Dense(128, activation='relu')(x)
        output_layer = layers.Dense(2, activation='softmax', kernel_regularizer=reg)(x)

        model = models.Model(input_layer, output_layer)
        model.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        self.model.summary()

    def make_model4(self):
        clf = svm.SVC(kernel='linear')

    def train(self, X, y, tissue_type):
        #x, v, y, w = train_test_split(x, y, test_size=0.2)
        reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-5)
        #xx = np.repeat(x, 9, axis=0)
        #yy = np.repeat(y, 9, axis=0)
        #xa = np.append(x, xx, axis=0)
        #ya = np.append(y, yy, axis=0)
        #print(xa.shape, ya.shape)
        #for i in range(3):
            #print("Episode:", i+1)
            #for j in range(len(xa)):
            #    np.random.shuffle(xa[j])
            #xx = x + (np.random.normal(0, 0.1, size=x.shape) * x)
            #print('max x:', np.max(xx), ' mean x:', np.mean(xx), ' min x:', np.min(xx), '\n')
        #    history = self.model.fit(x, y, epochs=i+1, initial_epoch=i, verbose=1, batch_size=self.batch_size, validation_data=(v, w), callbacks=[reduce])
        #    for j in range(len(x)):
        #        np.random.shuffle(x[j])
        #return history
        X, y = shuffle(X, y)
        es = EarlyStopping(monitor='val_loss', mode='min', patience=50)
        kf = KFold(3)
        aucs = []
        results = []
        cms = []
        for train_index, test_index in kf.split(X):
            self.make_model(X.shape)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train = np.concatenate([X_train for _ in range(6)])
            y_train = np.concatenate([y_train for _ in range(6)])
            X_train, y_train = shuffle(X_train, y_train)
            print("x shape:", X_train.shape)
            print("y shape:", y_train.shape)
            for i in range(50):
                history = self.model.fit(X_train, y_train, epochs=i+1, initial_epoch=i, verbose=1, batch_size=self.batch_size, validation_data=(X_test, y_test), callbacks=[es, reduce])
                for j in range(len(X_train)):
                    np.random.shuffle(X_train[j])
            results.append(self.model.evaluate(X_test, y_test, verbose=0))
            cms.append(confusion_matrix(y_test, np.argmax(self.model.predict(X_test), axis=-1)))
            fpr, tpr, _ = roc_curve(1-y_test, self.model.predict(X_test)[:, 0])
            precision, recall, _ = precision_recall_curve(1-y_test, self.model.predict(X_test)[:, 0])
            roc_auc = roc_auc_score(1-y_test, self.model.predict(X_test)[:, 0])
            pr_auc = auc(recall, precision)
            plt.plot(fpr, tpr)
            aucs.append(roc_auc)
        plt.plot(np.arange(len(y_test))/len(y_test), np.arange(len(y_test))/len(y_test), linestyle='--')
        plt.plot(fpr, tpr)
        plt.title("AUC: " + str(round(roc_auc, 4)))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig('./deployment_models2/' + tissue_type + '_' + self.model_name +'_Roc_Curve.png')
        plt.clf()
        #plt.show()
        plt.plot(np.arange(len(y_test))/len(y_test), np.zeros(len(y_test)), linestyle='--')
        plt.plot(recall, precision, marker='.')
        plt.title("AUC: " + str(round(pr_auc, 4)))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig('./deployment_models2/' + tissue_type +  '_' + self.model_name + '_Precision-Recall.png')
        plt.clf()
        #plt.show()


        #self.model.save_weights("./deployment_models/{}_weights.h5".format("ConvLSTM"))
        with open("./deployment_models2/{}_{}_ConvLSTM.txt".format(tissue_type, self.model_name), 'w') as f:
            f.write("Results\n\n")
            f.write("Evaluation loss and accuracy:\n")
            f.write(str(results))
            f.write("\n\nConfusion matrix\n")
            f.write(str(cms))

def create_dataset(crnn, tissue_type, model_name, diseased_dir, non_diseased_dir):
    df, ndf = crnn.extract_features(diseased_dir, non_diseased_dir)
    x_train, y_train = crnn.make_data(df, ndf)
    with open('x_{}_{}.pkl'.format(tissue_type, model_name), 'wb') as f:
        pickle.dump(x_train, f)
    with open('y_{}_{}.pkl'.format(tissue_type, model_name), 'wb') as f:
        pickle.dump(y_train, f)

def get_dataset(tissue_type, model_name):
    with open('x_{}_{}.pkl'.format(tissue_type, model_name), 'rb') as f:
        x_train = pickle.load(f)
    with open('y_{}_{}.pkl'.format(tissue_type, model_name), 'rb') as f:
        y_train = pickle.load(f)
    return (x_train, y_train)


if __name__ == "__main__":

    batch_size = 32
    tissue_type = sys.argv[1]
    model_num = int(sys.argv[3])
    diseased_dir = '../data/research/images/' + tissue_type + '/diseased/'
    non_diseased_dir = '../data/research/images/' + tissue_type + '/non_diseased/'

    CRNN = ConvLSTM(model_num, batch_size)
    if model_num == 1:
        CRNN.make_conv_base("experiment3/{}_weights.h5".format(tissue_type))
    if int(sys.argv[2]):
        create_dataset(CRNN, tissue_type, model_names[model_num], diseased_dir, non_diseased_dir)
    x_train, y_train = get_dataset(tissue_type, model_names[model_num])

    CRNN.make_model2(x_train.shape)
    print('\nx_shape:', x_train.shape, '\ny_shape:', y_train.shape)
    history = CRNN.train(x_train, y_train, tissue_type)
    print(history.history)
