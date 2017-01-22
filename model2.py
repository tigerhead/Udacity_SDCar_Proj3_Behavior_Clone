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
import cv2
#load conifig
import bcConfig as config
#Image preprocess
from imageProcess import preprocessImage2, augment_brightness_camera_images, trans_image


#Load config
log_file = config.DRIVE_LOG_FILE
image_path = config.DATA_FILE_PATH
image_channel = config.ORG_IMAGE_CHANNEL
image_h = config.ORG_IMAGE_HEIGHT
image_w = config.ORG_IMAGE_WIDTH
in_image_h = config.CORP_IMAGE_HEIGHT2
in_image_w = config.CORP_IMAGE_WIDTH2

#Parse drive log file
input = np.genfromtxt (log_file, dtype=None, delimiter=",")
input = input[1:]
print("Shape: ", input.shape)
print('Sample data: ', input[0])

#Training image file path
X_train_files = input[: ,[0, 1, 2 ]]
#Train Steering data
y_train = (input[:,3 ]).astype(np.float32)

print('Sample data: ', X_train_files[0])
print('Sample data: ', y_train[0])
print('Sample data: ', X_train_files[-100])
print('Sample data: ', y_train[-100])
print('Train files Shape: ', X_train_files.shape)
print('Train label shape: ', y_train.shape)



# Shuffle train data and label
X_train_files, y_train = shuffle(X_train_files, y_train)


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


#Load and preprocess image data
def preprocess_image_file_train(x, y):
    # Randomly load center, left or right image
    i_lrc = np.random.randint(3)
    if (i_lrc == 0):
        path_file = x[1].strip()
        shift_ang = .25
    if (i_lrc == 1):
        path_file = x[0].strip()
        shift_ang = 0.
    if (i_lrc == 2):
        path_file = x[2].strip()
        shift_ang = -.25
    y_steer = y + shift_ang
    image = None
    y_stee = None
    try:
        image = cv2.imread(image_path + get_file_name(path_file)).astype(np.float32)
        #Convert BGR to RGB
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # Randomly shift image
        image,y_steer = trans_image(image,y_steer,100)
        #Randomly brighten or darken image
        image = augment_brightness_camera_images(image)
        #Corp, resize and normalize
        image = preprocessImage2(image)
        image = np.array(image)
        #Flip image and steering
        ind_flip = np.random.randint(2)
        if ind_flip==0:
            image = cv2.flip(image,1)
            y_steer = -y_steer
    except IOError as e:
                    print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    return image,y_steer

pr_threshold = 0.9
def generate_data_2(path, X, y,batch_size = 64):

    while 1:
        X, y = shuffle(X, y)
        num_examples = X.shape[0]
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            X_batch = np.zeros(shape = (batch_size, in_image_h, in_image_w, image_channel ), dtype = np.float32)
            y_batch = np.zeros(shape = (batch_size, 1), dtype = np.float32)
            i = 0
            for img_f_name, steering in zip(X[offset: end], y[offset: end]):
                keep_pr = 0
                #x,y = preprocess_image_file_train(line_data)
                while keep_pr == 0:
                    X_ln, y_ln = preprocess_image_file_train(img_f_name, steering)
                    pr_unif = np.random
                    if abs(y_ln)<0.1:
                        pr_val = np.random.uniform()
                        if pr_val>pr_threshold:
                            keep_pr = 1
                    else:
                        keep_pr = 1
                X_batch[i] = X_ln
                y_batch[i] = y_ln
                i+=1
            yield X_batch, y_batch


def generate_data(path, X, y, batch_size):
    X_batch = np.zeros(shape = (batch_size, in_image_h, in_image_w, image_channel ), dtype = np.float32)
    y_batch = np.zeros(shape = (batch_size, 1), dtype = np.float32)
    while 1:
            for i_batch in range(batch_size):
                i_line = np.random.randint(len(X))
                img_f_name = X[i_line]
                steering = y[i_line]
                X_ln = None
                y_ln = None
                try:
                    keep_pr = 0
                    while keep_pr == 0:
                        X_ln, y_ln = preprocess_image_file_train(img_f_name, steering)
                        pr_unif = np.random
                        if abs(y_ln)<0.1:
                            pr_val = np.random.uniform()
                            if pr_val>pr_threshold:
                                keep_pr = 1
                        else:
                            keep_pr = 1
                    X_batch[i_batch] = X_ln
                    y_batch[i_batch] = y_ln
                except IOError as e:
                    print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')


            yield X_batch, y_batch


# convolution kernel size
input_shape = (in_image_h, in_image_w, image_channel)
pool_size = (2, 2)

#Nvidia model
def nvidia_model():
    model = Sequential()
    model.add(Convolution2D(3, 1, 1, border_mode='valid', input_shape=input_shape))
    model.add(Convolution2D(24, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Convolution2D(36, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))
    model.add(Convolution2D(48, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Activation('relu'))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.summary()
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

    #Set check point and path to save weights for each epoch
    checkpoint_path="models_2_01/model_{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False, save_weights_only=True, mode='auto')
    #Using generator to handle large dataset
    #for regression problem, use mean squre error as validation matrics
    model.fit_generator(generator=generate_data(image_path, X_train_files, y_train, batch_size),
                    samples_per_epoch = ((X_train_files.shape[0] - X_train_files.shape[0] % batch_size) * 2),
                    nb_epoch=nb_epoch,
                    verbose=2,
                    callbacks=[checkpoint]
                    #validation_data = generate_data_2(image_path, X_val_files, y_val, 50),
                    #nb_val_samples= (X_val_files.shape[0]- X_val_files.shape[0]%50)
                    )
    # pred = model.predict_generator(generate_data_2(image_path, X_val_files, y_val, batch_size), val_samples = X_val_files.shape[0]- X_val_files.shape[0]%batch_size)

    #for y_pred, y_target in zip(pred[0: 30], y_val[0: 30]):
    #   print(y_pred, y_target)

    #Save model and weights
    model_str =model.to_json()
    json_file = 'model2.json'
    weights_file = 'model2.h5'
    with open(json_file, 'w') as outfile:
        outfile.write(model_str)
    model.save_weights(weights_file)


