import os

from keras.models import Model
from keras.layers import (Input, Dense, Flatten)
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.utils import get_file

import keras.backend as K

from kerastools.imagerecon import ImageRecognition

class Vgg(ImageRecognition):
    '''
        A Python implementation of the VGG 16 Imagenet model with added average
        pooling layers. Extends ImageRecon.
    '''

    def __init__(self, layers = 16, include_top = True, weights = 'imagenet',
                 input_tensor = None, input_shape = None,
                 classes = 1000, pooling = 'avg', subtract_mean = False):
        # Intialise parent class
        super().__init__()
        
        if weights not in {'imagenet', None}:
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization) or `imagenet` '
                             '(pre-training on ImageNet).')
            
        if weights == 'imagenet' and include_top and classes != 1000:
            raise ValueError('If using `weights` as imagenet with `include_top`'
                             ' as true, `classes` should be 1000')
            
        self.PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/'
        
        if layers == 16:
            self.TF_WEIGHTS = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
            self.TF_WEIGHTS_NO_TOP = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        elif layers == 19:
            self.TF_WEIGHTS = 'vgg19_weights_tf_dim_ordering_tf_kernels.h5'
            self.TF_WEIGHTS_NO_TOP = 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
        else:
            raise ValueError('The number of layers should be one of (16, 19) '
                             'for VGG16 or VGG19 style networks.')

        # Create model
        self.create(layers = layers, include_top = include_top,
                    weights = weights, input_tensor = input_tensor,
                    input_shape = input_shape, classes = classes,
                    pooling = pooling, subtract_mean = subtract_mean)
        

    def conv_block(self, in_tensor, layers, filters, pooling, name):
        '''
            Adds a given number of zero padding and convolution layers to a
            
            eras model, with an avg pooling layer at the end of the layers.

            Args:
                in_tensor (tensor): Input tensor from previous layer

                layers (int):       The number of zero padded convolution 
                                    layers to be added to the model.

                filters (int):      The number of convolution filters to be 
                                    created for each layer.
                                    
                pooling (string):   One of 'avg' or max depending on type of 
                                    pooling

                name (str):         The name of the convolutional layer

            Returns:
                Tensor with the convolutional layer
        '''

        out_tensor = in_tensor

        # Add the number of layers specified
        for i in range(layers):

            # Add the convolution layer with the given number of filters, with a
            # 3 x 3 pixel window size (kernel), a 1 pixel stride and rectified
            # linear unit (relu) activation, which is simply max(0, x). (This
            # is our non-liniarity) and same (zero padding bordering).
            out_tensor = Conv2D(filters,
                                kernel_size = (3, 3),
                                strides = (1, 1),
                                activation = 'relu',
                                name = name + '_conv_' + str(i))(out_tensor)

        # Add a final max/avg pooling layer. We down scale by a factor of 2 x 2, and
        # move with a 2 x 2 stride.
        if pooling == 'avg':
            out_tensor = AveragePooling2D((2, 2),
                                          strides = (2, 2),
                                          name = name + '_avgpool')(out_tensor)
        
        elif pooling == 'max':
            out_tensor = MaxPooling2D((2, 2),
                                      strides = (2, 2),
                                      name = name + '_maxpool')(out_tensor)
        
        else:
            'Wrong specification of pooling layer'
        
        return out_tensor

    def create(self, layers = 16, include_top = True, weights = 'imagenet',
               input_tensor = None, input_shape = None, pooling = 'avg',
               classes = 1000, subtract_mean = False):
        '''
            Creates the VGG16 network architecture with average pooling and 
            loads the pretrained weights.

            Args:

                layers (int):           Number of total layers in model. One of
                                        (16, 19) for VGG16 or VGG19 style
                                        network.

                include_top (bool):     Specify wether or not to include the 
                                        last fully connected output layer. Will
                                        have to be specified manually if not.
                                        
                weights (string):       Should be either '`None` (random 
                                        initialization) or `imagenet` 
                                        (pre-training on ImageNet).
                                        
                input_tensor (tensor):  Optional provide a preprocessed tensor,
                                        eg. a lambda layer subtracting mean for
                                        each channel.
                
                input_shape (tupple):   User specified image shape
                
                classes (integer):      Number of classes in final layers, 1000
                                        if Imagenet is used as weights.
                                        
                pooling (string):       One of 'avg' or max depending on type of 
                                        pooling

                subtract_mean (bool):   Should the mean (Imagenet) of each
                                        channel be subtracted at initialisation?

            Returns:
                None
        '''

        # Determine proper input shape using the Keras utility to compute/validate 
        # an ImageNet model's input shape. Returns the proper input size as tupple
        input_shape = _obtain_input_shape(input_shape,
                                          default_size = 224,
                                          min_size = 48,
                                          data_format = K.image_data_format(),
                                          require_flatten = include_top)
        
        # Ensure correct shape if input tensor is provided.
        if input_tensor is None:
            img_input = Input(shape = input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor = input_tensor, shape = input_shape)
            else:
                img_input = input_tensor

        # Note, that we do not do the subtraction of mean in this code by,
        # default, as the pupose of this model is to be generative. So we want
        # to go back and forth between image preprocessing outside of the model 
        # scope
        if subtract_mean:
            img_input = Lambda(self.vgg_preprocess,
                            output_shape = size + (3,),
                            name = 'norm_layer')(img_input)


        # Add the zero padded convolutional layers (13 or 17)

        conv_layer_1 = self.conv_block(img_input, 2, 64, pooling = pooling, name = 'block_1')
        conv_layer_2 = self.conv_block(conv_layer_1, 2, 128, pooling = pooling, name = 'block_2')
        
        if layers == 16:
            
            conv_layer_3 = self.conv_block(conv_layer_2, 3, 256, pooling = pooling, name = 'block_3')
            conv_layer_4 = self.conv_block(conv_layer_3, 3, 512, pooling = pooling, name = 'block_4')
            x = self.conv_block(conv_layer_4, 3, 512, pooling = pooling, name = 'block_5')
        
        elif layers == 19:
            conv_layer_3 = self.conv_block(conv_layer_2, 4, 256, pooling = pooling, name = 'block_3')
            conv_layer_4 = self.conv_block(conv_layer_3, 4, 512, pooling = pooling, name = 'block_4')
            x = self.conv_block(conv_layer_4, 4, 512, pooling = pooling, name = 'block_5')

        # Add dense layer if specified for classification
        if include_top and pooling == 'avg':
            
            x = Flatten(name = 'flatten')(x)
            x = Dense(4096, activation = 'relu', name = 'fc1')(x)
            x = Dense(4096, activation = 'relu', name = 'fc2')(x)
            x = Dense(classes, activation = 'softmax', name = 'predictions')(x)
            
        elif include_top:
            x = Flatten(name = 'flatten')(x)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dropout(0.5)(x)
            x = Dense(4096, activation='relu', name='fc2')(x)
            x = Dropout(0.5)(x)
            x = Dense(classes, activation = 'softmax', name = 'predictions')(x)
            
        # Ensure that the model takes into account any potential predecessors 
        #of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input
            
        
        # Create model.
        self.model = Model(inputs, x, name = 'vgg' + str(layers) + pooling)

        # Finally load the pretrained weights and cache them
        if weights == 'imagenet':
            
            if K.image_dim_ordering() == 'tf':
                
                wname = self.TF_WEIGHTS if include_top else self.TF_WEIGHTS_NO_TOP
                weights_path = get_file(wname, self.PATH + wname, cache_subdir = 'models')
                
                self.model.load_weights(weights_path)
