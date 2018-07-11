"""
This module contains all the utility functions and links for vgg16.py.

Methods:
load_predict_image -- returns two arrays, one containing image data, 
                       the other, the corresponding image name.
process_input_image -- returns the resized and RGB converted image, 
                       to be used in load_predict_image.
"""


import cv2
import numpy as np
import os
from os.path import dirname


LABELS_FILE = 'rcmalli_vggface_labels_v1.npy'
WEIGHTS_FILE = 'vgg16_face_weights.h5'
PREDICT_IMAGE_FOLDER = 'predict_images'

#WEIGHTS_PATH = 'drive/vgg16/weights/' + WEIGHTS_FILE
WEIGHTS_PATH = dirname(os.getcwd())+"/weights/" + WEIGHTS_FILE
#LABELS_PATH = 'drive/vgg16/weights/rcmalli_vggface_labels_v1.npy'
LABELS_PATH = dirname(os.getcwd())+"/weights/" + LABELS_FILE
#PREDICT_IMAGE_PATH = 'drive/vgg16/predict_images'
PREDICT_IMAGE_PATH = dirname(os.getcwd()) + '/' + PREDICT_IMAGE_FOLDER

WEIGHTS_LINK = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5'
LABELS_LINK = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels_v1.npy'

def load_predict_image():
    """
    Return processed images in predict_images in an numpy array format.
    """
    path = PREDICT_IMAGE_PATH
    images = os.listdir(path)                           # Create list of images name
    array_rows = len(images)                            # Number of images in the directory.

    image_arr = []                                      # Initialize image_array to store images
    image_names = []                                    # Initialize array to store image names

    print("Processing images from {}...".format(PREDICT_IMAGE_FOLDER))
    for i, image in enumerate(images):                  # Store image and image names in the arrays vars.
        if image.endswith('jpg'):
            image_arr.append(process_input_image(image_name=image, image_path=path))
            image_names.append(image)



    image_arr = np.asarray(image_arr, dtype=np.float32) # Convert to float32 for appropriate keras input dtype.
    
    
    #assert(image_arr.shape == (array_rows, 224, 224, 3))# Sanity check.
    print("Images for prediction loaded.")
    return image_arr, image_names


def process_input_image(image_name, image_path, flatten = False):
    """
    Resize and convert image to RGB.

    Keyword Arguments:
    image_name -- name of image. (e.g. 'man.jpg')
    image_path -- path to the image directory
    

    """
    img_handle = os.path.join(image_path, image_name)  # Obtain handle for image.
    print(img_handle)
    img = cv2.imread(img_handle)                # Read image as array, in BGR
    print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR image to RGB
    img = cv2.resize(img, (224, 224))           # Resize image to 244, 244, 3
    
    if flatten:
        img = img.reshape(1, flatten_length)  # Flatten image to feed into NN as features

    return img

def load_labels():
    """
    Returns reshaped labels from .npy file.
    Number of classes = 2622.
    """
    path = LABELS_PATH
    if os.path.exists(path): # Check if label file already exists
        print("{} file found.".format(LABELS_FILE))
    else:                           # If not, download from rcmalli's link.
        print("Downloading labels from rcmalli...")
        path = get_file(fname = LABELS_PATH, origin = 
                                LABELS_LINK)
    print("Loading labels...")
    labels = np.load(path) # Load labels from .npy
    labels = labels.reshape(labels.shape[0], 1)

    print("Labels loaded.")

    return labels

def load_weights(model):
    # Load weights
    path = WEIGHTS_PATH
    if os.path.exists(path):
        print("{} file found.".format(WEIGHTS_FILE))
    else:
        print("Downloading weights from rcmalli...")
        path = get_file(fname = WEIGHTS_PATH, origin =
                                WEIGHTS_LINK)
    print("Loading weights into model...")
    model.load_weights(filepath=path, by_name=True)    # Use keras load_weights to load weights from rcmalli
    print("Weights loaded to model.")
    
    return model


