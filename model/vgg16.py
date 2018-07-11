"""
    This module makes prediction for the images using vgg16 architecture.
    Weights used are transferred form rcmalli's vgg16face.
    Images for prediction is stored in 'predict_images' directory.
    
    Prints the predicted label for each image in the directory with highest probability.

    Note: Model uses tensorflow as backend on default, modifiable via ~jackh/.keras/keras.json

    Related modules:
    model.py
    utils.py
    rcmalli_utils.py
"""


import keras
import keras.backend as K
from keras.utils.data_utils import get_file
from keras.preprocessing import image

import numpy as np
import os
from os.path import dirname
import argparse

# Self written modules
import utils
import rcmalli_utils
from model import create_model
import time



def main():
    start_time = time.time()
    # Create model
    K.tensorflow_backend._get_available_gpus()  # Check if it is using tensorflow gpu
    start_create = time.time()
    model = create_model()                      # Create the vgg16 model
    end_create = time.time()
    print("Time elapsed creating model: {} seconds".format(end_create - start_create))

    # Load weights into model
    start_weights = time.time()
    model = utils.load_weights(model)
    end_weights = time.time()
    print("Time elapsed loading weights: {} seconds".format(end_weights-start_weights))

    # Load predict images
    x, image_names = utils.load_predict_image()

    # Load labels
    labels = utils.load_labels()   

    # My own method: Preprocess input using OpenCV library. #
    start_predictions = time.time()
    predictions = model.predict(x=x)    # Use model to predict
    end_predictions = time.time()
    print("Time elapsed to predict with model: {} seconds.".format(end_predictions-start_predictions))
    label = ''
    confidence = 0
    for i in range(len(predictions)):
        label = labels[np.argmax(predictions[i])]
        confidence = np.max(predictions[i])*100
        print("The prediction for image {} label: {} with confidence {}"
              .format(image_names[i], label, confidence))
    end_time = time.time()
    print("Total time elapsed: {} seconds.".format(end_time - start_time))


    ## From rcmalli's utils: Preprocessing input using keras and numpy. #
    #img = image.load_img("drive/vgg16/predict_images/adina_porter.jpg", target_size=(224, 224))
    #x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    #x = rcmalli_utils.preprocess_input(x, version=2) # or version=2
    #preds = model.predict(x)
    #print('Predicted:', rcmalli_utils.decode_predictions(preds))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--colab", type = bool, help = 'will this file be run on google colab or locally?')
    args = parser.parse_args()
    main()
