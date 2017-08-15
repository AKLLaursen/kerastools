from keras.layers import Activation, BatchNormalization
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.applications.resnet50 import identity_block, conv_block

from .imagerecon import ImageRecognition

class Resnet50(ImageRecognition):
    """
        A Python implementation of the Resnet 50 Imagenet model. Extends 
        ImageRecon.
    """

    def __init__(self, size = (224,224), include_top = True):
        self.create(size, include_top)


    def create(self, size, include_top):
        """
            Creates the Resnet50 network architecture and loads the pretrained
            weights.

            Args:
                size (tuple):           Pixel size of input images. Default for 
                                        Resnet50 is (224 x 224).

                include_top (bool):     Specify wether or not to include the 
                                        last fully connected output layer. Will
                                        have to be specified manually if not.

            Returns:
                None
        """

        # Define input layer (hard coded input shape, as Resnet 50 was trained 
        # on 224 x 224 images
        inputs = Input(shape = (3,) + size)

        # Define first layer stack of Resnet 50.

        # Preprocess data by subtracting mean from imagenet and rearraging 
        # channels
        x = Lambda(self.vgg_preprocess)(inputs)

        # Pad with zeros
        x = ZeroPadding2D((3, 3))(x)

        # Define 7x7 convolutional layer with 2x2 stride
        x = Conv2D(64,
                   kernel_size = (7, 7),
                   strides = (2, 2),
                   name = 'conv1')(x)

        # Utalise batchnorm
        x = BatchNormalization(axis = 1, name = 'bn_conv1')(x)

        # Apply relu activation function (note after batchnorm!)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides = (2, 2))(x)

        # Define convolutional block and image layers
        x = conv_block(x, 3, [64, 64, 256], stage = 2, block = 'a',
                       strides = (1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage = 2, block = 'b')
        x = identity_block(x, 3, [64, 64, 256], stage = 2, block = 'c')

        x = conv_block(x, 3, [128, 128, 512], stage = 3, block = 'a')

        # Three layer identity block
        for n in ['b','c','d']:
            x = identity_block(x, 3, [128, 128, 512], stage = 3, block = n)

        x = conv_block(x, 3, [256, 256, 1024], stage = 4, block = 'a')

        # Five layer identity block
        for n in ['b','c','d', 'e', 'f']:
            x = identity_block(x, 3, [256, 256, 1024], stage = 4, block = n)

        x = conv_block(x, 3, [512, 512, 2048], stage = 5, block = 'a')
        x = identity_block(x, 3, [512, 512, 2048], stage = 5, block = 'b')
        x = identity_block(x, 3, [512, 512, 2048], stage = 5, block = 'c')

        # We then specify the output layer and the pretrained weights based on 
        # wether or not the denselayers are to be included
        if include_top:
            x = AveragePooling2D((7, 7), name = 'avg_pool')(x)
            x = Flatten()(x)
            x = Dense(1000, activation = 'softmax', name = 'fc1000')(x)

            fname = 'resnet50.h5'

        else:
            fname = 'resnet_nt.h5'

        self.img_input = inputs

        # Initialise model
        self.model = Model(self.img_input, x)

        #Convert all convolution kernels in a model from TensorFlow to Theano
        convert_all_kernels_in_model(self.model)

        # Finally load the pretrained weights and cache them
        self.model.load_weights(get_file(fname,
                                         self.FILE_PATH + fname,
                                         cache_subdir = 'models'))
