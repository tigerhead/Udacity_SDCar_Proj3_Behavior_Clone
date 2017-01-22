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
#load conifig
import bcConfig as config
#image preprocess
from imageProcess import preprocessImage
import cv2

#Load config data from config file
log_file = config.DRIVE_LOG_FILE
image_path = config.DATA_FILE_PATH
image_channel = config.ORG_IMAGE_CHANNEL
image_h = config.ORG_IMAGE_HEIGHT
image_w = config.ORG_IMAGE_WIDTH
in_image_h = config.CORP_IMAGE_HEIGHT
in_image_w = config.CORP_IMAGE_WIDTH
data_file_path = config.DATA_FILE_PATH

#Parse drive log file
input = np.genfromtxt (log_file, dtype=None, delimiter=",")
input = input[1:]
print("Shape: ", input.shape)
print('Sample data: ', input[0])

#Center image
X_train_files = input[: ,0]
#Left image
X_train_files_l = input[: ,1]
#Right image
X_train_files_r = input[: ,2]
y_train_c = (input[:,3 ]).astype(np.float32)

#Left and right image steering adjustment
left_steering_correction = 0.27
right_steering_correction = -0.27

# Add left image to train data set
X_train_files= np.append(X_train_files, X_train_files_l)
#Add adjusted left image steering
y_train = np.append(y_train_c, y_train_c + left_steering_correction)

# Add left image to train data set
X_train_files= np.append(X_train_files, X_train_files_r)
#Add adjusted left image steering
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

# Shuffle data
X_train_files, y_train = shuffle(X_train_files, y_train)

#Split dataset and labels to traning and validation dataset, labels using
def split_data(X, y, t_size = 0.1):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= t_size, random_state=42, stratify=None)
    return X_train, X_test, y_train, y_test
#Split test and validation set. Since the validtion set doesn't really help to train the model,
# the split is not used in the end.'
#X_train_files, X_val_files, y_train, y_val = split_data(X_train_files, y_train)



print('Train Shape: ', X_train_files.shape)
print('Train label shape: ', y_train.shape)

#Get teh iamge file name
def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def get_file_name(s):
    return find_between(str(s), "IMG/", "'")



#Data generator, genrate input batch

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
                    #Randomly flip image
                    ind_flip = np.random.randint(2)
                    if ind_flip==0:
                        img = cv2.flip(img,1)
                        steering = -steering
                    #Resize and normalize image
                    X_batch[i_batch] = preprocessImage(img.astype(np.float32))
                    y_batch[i_batch] = steering
                except IOError as e:
                    print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

            yield X_batch, y_batch


#Input image shape
input_shape = (in_image_h, in_image_w, image_channel)
# Pool size
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
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(20))
    model.add(Dense(1))
    return model


if __name__ == '__main__':
    K.clear_session()
    model = nvidia_model()
    model.summary()
    batch_size = 64
    nb_epoch = 8
    model.compile(loss='mse',
              optimizer=Adam(lr=0.0001),
              metrics=['mse']
             )

     #Set check point to save weights for each epoch
    checkpoint_path="models_1_01/model_{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False, save_weights_only=True, mode='auto')
    #Using generator to handle large dataset
    #for regression problem, use mean squre error as validation matrics
    model.fit_generator(generator=generate_data(image_path, X_train_files, y_train, batch_size),
                    samples_per_epoch = (X_train_files.shape[0] - X_train_files.shape[0] % batch_size),
                    nb_epoch=nb_epoch,
                    verbose=2,
                    callbacks=[checkpoint]
                    #validation_data = generate_data_2(image_path, X_val_files, y_val, 50),
                    #nb_val_samples= (X_val_files.shape[0]- X_val_files.shape[0]%50
                    )

    # Save model and final weights to file
    model_str =model.to_json()
    json_file = 'model.json'
    weights_file = 'model.h5'
    with open(json_file, 'w') as outfile:
        outfile.write(model_str)
    model.save_weights(weights_file)


