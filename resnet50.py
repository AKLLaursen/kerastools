import os

from keras.models import Model
from keras.layers import Input, Lambda, Flatten, Activation, BatchNormalization, Dense, add
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.utils.data_utils import get_file

from .imagerecon import ImageRecognition

class Resnet50(ImageRecognition):
    """
        A Python implementation of the Resnet 50 Imagenet model. Extends 
        ImageRecon.
    """

    def __init__(self, size = (224, 224), include_top = True):
        super().__init__()
        self.create(size, include_top)
        
        
    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        """
            A stack of three convolutional layers with batchnorm with the input identity 
            added before the final activation.
            
            Args:
                input_tensor (tensor): The output of a previous layer.
                kernel_size (int):     Default 3, the kernel size of middle conv layer at main path.
                filters (list):        A list of integers, being the filters of 3 conv layer at main path.
                stage (int):           The current stage label, used for generating layer names.
                block (str):           Eg. 'a','b'..., the current block label, used for generating layer names
                
            Returns:
                Output tensor for the block.
        """
    
        # Specify filters from list
        filters1, filters2, filters3 = filters
        
        # We use TensorFlow, so image last
        bn_axis = 3
        
        # Define base names to be used for each layer
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        x = Conv2D(filters1, (1, 1), name = conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis = bn_axis, name = bn_name_base + '2a')(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filters2, kernel_size, padding = 'same', name = conv_name_base + '2b')(x)
        x = BatchNormalization(axis = bn_axis, name = bn_name_base + '2b')(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filters3, (1, 1), name = conv_name_base + '2c')(x)
        x = BatchNormalization(axis = bn_axis, name = bn_name_base + '2c')(x)
        
        # Add input layer before final activation.
        x = add([x, input_tensor])
        x = Activation('relu')(x)
        
        return x


    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides = (2, 2)):
        """
            A stack of three convolutional layers with batchnorm with the input tensor fed into 
            a single CNN with batchnorm before being added to the final activation. Note that from 
            stage 3, the first conv layer at main path is with strides = (2, 2) And the shortcut 
            should have strides = (2, 2) as well. (I.e. we change the output dimension).
            
            Args:
                input_tensor (tensor): The output of a previous layer.
                kernel_size (int):     Default 3, the kernel size of middle conv layer at main path.
                filters (list):        A list of integers, being the filters of 3 conv layer at main path.
                stage (int):           The current stage label, used for generating layer names.
                block (str):           Eg. 'a','b'..., the current block label, used for generating layer names
                strides (tuple):       Default is (2, 2)
                
            Returns:
                Output tensor for the block.
                
    """
        # Specify filters from list
        filters1, filters2, filters3 = filters
        
        # We use TensorFlow, so image last
        bn_axis = 3
        
        # Define base names to be used for each layer
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        x = Conv2D(filters1, (1, 1), strides = strides, name = conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis = bn_axis, name = bn_name_base + '2a')(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filters2, kernel_size, padding = 'same', name = conv_name_base + '2b')(x)
        x = BatchNormalization(axis = bn_axis, name = bn_name_base + '2b')(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filters3, (1, 1), name = conv_name_base + '2c')(x)
        x = BatchNormalization(axis = bn_axis, name = bn_name_base + '2c')(x)
        
        # Feed input tensor into a new single CNN with max pooling
        shortcut = Conv2D(filters3, (1, 1), strides = strides, name = conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis = bn_axis, name = bn_name_base + '1')(shortcut)
        
        # Add input layer before final activation.
        x = add([x, shortcut])
        x = Activation('relu')(x)
        
        return x

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
        inputs = Input(shape = size + (3,))

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
        x = BatchNormalization(axis = 3, name = 'bn_conv1')(x)

        # Apply relu activation function (note after batchnorm!)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides = (2, 2))(x)

        # Define convolutional block and image layers
        x = self.conv_block(x, 3, [64, 64, 256], stage = 2, block = 'a', strides = (1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage = 2, block = 'b')
        x = self.identity_block(x, 3, [64, 64, 256], stage = 2, block = 'c')

        x = self.conv_block(x, 3, [128, 128, 512], stage = 3, block = 'a')

        # Three layer identity block
        for n in ['b','c','d']:
            x = self.identity_block(x, 3, [128, 128, 512], stage = 3, block = n)

        x = self.conv_block(x, 3, [256, 256, 1024], stage = 4, block = 'a')

        # Five layer identity block
        for n in ['b','c','d', 'e', 'f']:
            x = self.identity_block(x, 3, [256, 256, 1024], stage = 4, block = n)

        x = self.conv_block(x, 3, [512, 512, 2048], stage = 5, block = 'a')
        x = self.identity_block(x, 3, [512, 512, 2048], stage = 5, block = 'b')
        x = self.identity_block(x, 3, [512, 512, 2048], stage = 5, block = 'c')

        # We then specify the output layer and the pretrained weights based on 
        # wether or not the denselayers are to be included
        if include_top:
            
            x = AveragePooling2D((7, 7), name='avg_pool')(x)
            x = Flatten()(x)
            x = Dense(1000, activation = 'softmax', name = 'fc1000')(x)

            fname = 'resnet50.h5'

        else:
            
            fname = 'resnet50-no-top.h5'

        self.img_input = inputs

        # Initialise model
        self.model = Model(self.img_input, x)

        # Finally load the pretrained weights and cache them
        self.model.load_weights(os.path.join(os.path.dirname(__file__), self.WEIGHT_PATH + fname))
