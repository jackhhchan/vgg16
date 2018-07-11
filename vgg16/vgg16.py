import keras
import keras.backend as K
from keras.utils.data_utils import get_file

import numpy as np
import os
from os.path import dirname

# Self written module imports
import utils
from model import create_model

def main():
    K.tensorflow_backend._get_available_gpus()  # Check if it is using tensorflow gpu
    model = create_model()                      # Create the vgg16 model

    # Load weights
    try:
        path = os.path.join(dirname(os.getcwd()), 'weights\vgg16_face_weights.h5')
        if os.path.exists(path):
            print("vgg16_face_weights.h5 file found.")
            weights_path = path
        else:
            print("Downloading weights from rcmalli...")
            weights_path = get_file('vgg16_face_weights.h5', 
                                    'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5')
            print("Download complete.")
        
        print("Loading weights into model...")
        model.load_weights(filepath = weights_path, by_name = True)
    except Exception:
        print("weights not found in {}".format(weights_path))

    # Load predict images
    print("Loading images from 'x_predict'...")
    x = utils.load_predict_data()
    # Use model to predict
    model.predict(x)

if __name__ == '__main__':
    main()
