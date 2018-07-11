"""
This module contains all the methods relating to the CNN architecture 
used in vgg16.py. The architecture is VGG16, created using keras.

Methods:
create_model() -- returns keras model with vgg16 architecture
                  Naming conventions follow rcmalli's vgg16face.

"""


import keras
from keras.models import Model
from keras.layers import Dense, Activation, Conv2D, \
    MaxPooling2D, Dropout, Flatten
from keras import optimizers
from keras import losses


def create_model():
    """
    The VGG16 architecture uses:
    Convolutional filters = (3 x 3)
    Max Pooling filters = (2 x 2)
    Block 1 -- 2 x (64 x Conv filters) layers, 1 x Max Pooling 
    Block 2 -- 2 x (128 x Conv filters) layers, 1 x Max Pooling
    Block 3 -- 3 x (256 x Conv filters) layers, 1 x Max Pooling
    Block 4 -- 3 x (512 Conv filters) layers, 1 x Max Pooling
    Block 5 -- same as block 4
    Block 6 -- Fully Connected layers (2 x ReLu, 1 x Softmax)
               Softmax layer is the classification layer.
    
    Number of labels is 2622.
    """
    num_classes = 2622          # Number of classes used in rcmalli's vgg16face.

    # Input layer dimensions
    # specify input layer dimensions to be (224, 244, 3)
    img_input = keras.engine.input_layer.Input(shape=(224, 224, 3))

    # BLOCK 1 (2x CONV64, 1x Max Pooling)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same',
               activation='relu', name='conv1_1')(img_input)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='conv1_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)

    # BLOCK 2 (2x CONV128, 1x Max Pooling)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same',
               activation='relu', name='conv2_1')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same',
               activation='relu', name='conv2_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(x)

    # BLOCK 3 (2x CONV256, 1x Max Pooling)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same',
               activation='relu', name='conv3_1')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same',
               activation='relu', name='conv3_2')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same',
               activation='relu', name='conv3_3')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(x)

    # BLOCK 4 (3x CONV512, 1x Max Pooling)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same',
               activation='relu', name='conv4_1')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same',
               activation='relu', name='conv4_2')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same',
               activation='relu', name='conv4_3')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool4')(x)

    # BLOCK 5 (3x CONV512, 1x Max Pooling)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same',
               activation='relu', name='conv5_1')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same',
               activation='relu', name='conv5_2')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same',
               activation='relu', name='conv5_3')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5')(x)

    # Block 6 (2 x FC4096, 1x Softmax)
    x = Flatten()(x)
    x = Dense(units=4096, name = 'fc6')(x)
    x = Activation(activation = 'relu', name = 'fc6/relu')(x)
    x = Dense(units=4096, name = 'fc7')(x)
    x = Activation(activation = 'relu', name = 'fc7/relu')(x)
    x = Dense(units=num_classes, name = 'fc8')(x)
    predictions = Activation(activation='softmax', name = 'fc8/softmax')(x)


    
    model = Model(inputs=img_input, outputs=predictions, name='vggface_vgg16') # Construct layers of the model
    
    model.summary() # Print model architecture summary


    return model


def train_model(model, batch_size, epoch):

    # Instantiate optimizer & loss function
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=False)
    cross_entropy = losses.categorical_crossentropy
    # Compile model
    model.compile(optimizers=sgd, loss=cross_entropy)
    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=32)     # x_train, y_train are numpy arrays.


if __name__ == '__main__':
    main()
