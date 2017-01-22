import numpy as np
from sklearn.model_selection import train_test_split
import math
from sklearn.utils import shuffle
import matplotlib.image as mpimg
import os
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import bcConfig as config
#image preprocess
from imageProcess import preprocessImage
import cv2


log_file = config.DRIVE_LOG_FILE
image_path = config.DATA_FILE_PATH
image_channel = config.ORG_IMAGE_CHANNEL
image_h = config.ORG_IMAGE_HEIGHT
image_w = config.ORG_IMAGE_WIDTH
in_image_h = config.CORP_IMAGE_HEIGHT
in_image_w = config.CORP_IMAGE_WIDTH
data_file_path = config.DATA_FILE_PATH


input = np.genfromtxt (log_file, dtype=None, delimiter=",")
input = input[1:]
print("Shape: ", input.shape)
print('Sample data: ', input[0])
#X_train_files = input[: ,[0, 1, 2 ]]
X_train_files = input[: ,0]
X_train_files_l = input[: ,1]
X_train_files_r = input[: ,2]
y_train_c = (input[:,3 ]).astype(np.float32)
left_steering_correction = 0.27
right_steering_correction = -0.27
X_train_files= np.append(X_train_files, X_train_files_l)
y_train = np.append(y_train_c, y_train_c + left_steering_correction)
X_train_files= np.append(X_train_files, X_train_files_r)
y_train = np.append(y_train, y_train_c + right_steering_correction)

print('Sample data: ', X_train_files[0])
print('Sample data: ', y_train[0])
print('Sample data: ', X_train_files[-100])
print('Sample data: ', y_train[-100])
print('Train files Shape: ', X_train_files.shape)
print('Train label shape: ', y_train.shape)

print('Sample data: ', X_train_files[0])
print('Sample data: ', y_train[0])
print('Train files Shape: ', X_train_files.shape)
print('Train label shape: ', y_train.shape)

# TODO: Use `train_test_split` here.
X_train_files, y_train = shuffle(X_train_files, y_train)

#Split dataset and labels to traning and validation dataset, labels using
def split_data(X, y, t_size = 0.1):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= t_size, random_state=42, stratify=None)
    return X_train, X_test, y_train, y_test

#X_train_files, X_val_files, y_train, y_val = split_data(X_train_files, y_train)
print('Train Shape: ', X_train_files.shape)
print('Train label shape: ', y_train.shape)
#print('Valid Shape: ', X_val_files.shape)
#print('Valid label shape: ', y_val.shape)

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def get_file_name(s):
    return find_between(str(s), "IMG/", "'")






# TODO: Implement data normalization here.
def normalize(x):
    image_depth = 255.0
    x= cv2.resize(x,(in_image_w, in_image_h),  interpolation=cv2.INTER_AREA)
    return (x - image_depth/2)/image_depth


def generate_data_2(path, X, y, batch_size):
    while 1:
        X, y = shuffle(X, y)
        num_examples = X.shape[0]
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            X_batch = np.zeros(shape = (batch_size, image_h, image_w, image_channel ), dtype = np.float32)
            y_batch = np.zeros(shape = (batch_size, 1), dtype = np.float32)
            i = 0
            for img_f_name, steering in zip(X[offset: end], y[offset: end]):
                image_file = os.path.join(path, get_file_name(img_f_name))
                try:
                    img = mpimg.imread(image_file)
                    X_batch[i] = preprocessImage(img.astype(np.float32))
                    y_batch[i] = steering
                except IOError as e:
                    print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
                i+=1
            yield X_batch, y_batch


def generate_data(path, X, y, batch_size):
    X_batch = np.zeros(shape = (batch_size, in_image_h, in_image_w, image_channel ), dtype = np.float32)
    y_batch = np.zeros(shape = (batch_size, 1), dtype = np.float32)
    while 1:
            for i_batch in range(batch_size):
                i_line = np.random.randint(len(X))
                img_f_name = X[i_line]
                image_file = os.path.join(path, get_file_name(img_f_name))
                steering = y[i_line]
                try:
                    img = mpimg.imread(image_file)
                    ind_flip = np.random.randint(2)
                    if ind_flip==0:
                        img = cv2.flip(img,1)
                        steering = -steering
                    X_batch[i_batch] = normalize(img.astype(np.float32))
                    y_batch[i_batch] = steering
                except IOError as e:
                    print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

            yield X_batch, y_batch


# convolution kernel size
input_shape = (in_image_h, in_image_w, image_channel)
pool_size = (2, 2)

#Nvidia model
def nvidia_model():
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, border_mode='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Convolution2D(36, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))
    model.add(Convolution2D(48, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(20))
    model.add(Dense(1))
    return model




if __name__ == '__main__':
    K.clear_session()
    model = nvidia_model()
    model.summary()
    batch_size = 64
    nb_epoch = 10
    model.compile(loss='mse',
              optimizer=Adam(lr=0.0001),
              metrics=['mse']
             )

    #Using generator to handle large dataset
    #for regression problem, use mean squre error as validation matrics
    checkpoint_path="models_1_01/model_{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False, save_weights_only=True, mode='auto')
    model.fit_generator(generator=generate_data(image_path, X_train_files, y_train, batch_size),
                    samples_per_epoch = (X_train_files.shape[0] - X_train_files.shape[0] % batch_size),
                    nb_epoch=nb_epoch,
                    verbose=2,
                    callbacks=[checkpoint]
                    #validation_data = generate_data_2(image_path, X_val_files, y_val, 50),
                    #nb_val_samples= (X_val_files.shape[0]- X_val_files.shape[0]%50)
                    )
   # pred = model.predict_generator(generate_data_2(image_path, X_val_files, y_val, batch_size), val_samples = X_val_files.shape[0]- X_val_files.shape[0]%batch_size)

   # for y_pred, y_target in zip(pred[0: 30], y_val[0: 30]):
     #   print(y_pred, y_target)

    model_str =model.to_json()
    json_file = 'model.json'
    weights_file = 'model.h5'
    with open(json_file, 'w') as outfile:
        outfile.write(model_str)
    model.save_weights(weights_file)


