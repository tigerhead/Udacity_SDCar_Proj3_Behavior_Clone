import cv2
import numpy as np
import bcConfig as config
import math

image_h = config.ORG_IMAGE_HEIGHT
image_w = config.ORG_IMAGE_WIDTH
in_image_h = config.CORP_IMAGE_HEIGHT
in_image_w = config.CORP_IMAGE_WIDTH
in_image_h2 = config.CORP_IMAGE_HEIGHT2
in_image_w2 = config.CORP_IMAGE_WIDTH2



#Changing brightness or darken to simulate day and night conditions
def augment_brightness_camera_images(image):

    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


#Horizontal and vertical shifts

def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(image_w,image_h))

    return image_tr,steer_ang



#Shadow augmentation

def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)


def normalize(x):
    image_depth = 255.0
    return (x - image_depth/2)/image_depth


def preprocessImage(x):
    image_depth = 255.0
    x= cv2.resize(x,(in_image_w, in_image_h),  interpolation=cv2.INTER_AREA)
    return normalize(x)


def preprocessImage2(image):
    shape = image.shape
    # note: numpy arrays are (row, col)!
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(in_image_h2,in_image_w2),  interpolation=cv2.INTER_AREA)
    #image = image/255.-.5
    image = normalize(image)
    return image