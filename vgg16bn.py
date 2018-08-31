import os

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (Input, Dense, BatchNormalization, Flatten, Dropout,
                          Lambda)
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.python.keras.utils import get_file

from kerastools.imagerecon import ImageRecognition

class Vgg16(ImageRecognition):
    '''
        A Python implementation of the VGG 16 Imagenet model. Extends 
        ImageRecon.
    '''

    def __init__(self, size = (224, 224), use_batchnorm = False,
                 include_top = True):
        # Intialise parent class
        super().__init__()
        # Create model
        self.create(size, use_batchnorm, include_top)
        

    def conv_block(self, in_tensor, layers, filters, name):
        '''
            Adds a given number of zero padding and convolution layers to a
            Keras model, with a max pooling layer at the end of the layers.

            Args:
                in_tensor (tensor): Input tensor from previous layer

                layers (int):       The number of zero padded convolution 
                                    layers to be added to the model.

                filters (int):      The number of convolution filters to be 
                                    created for each layer.

                name (str):         The name of the convolutional layer

            Returns:
                Tensor with the convolutional layer
        '''

        out_tensor = in_tensor

        # Add the number of layers specified
        for i in range(layers):

            # For each convolution layer we add a 1 pixel zero padded border.
            # We do this to preserve the dimensions of the image when doing the
            # convolution. (We use (1, 1) strides in the convolution layer,
            # which calls for a (1, 1) zero padding to preserve the dimensions).
            out_tensor = ZeroPadding2D((1, 1))(out_tensor)

            # Add the convolution layer with the given number of filters, with a
            # 3 x 3 pixel window size (kernel), a 1 pixel stride and rectified
            # linear unit (relu) activation, which is simply max(0, x). (This
            # is our non-liniarity)
            out_tensor = Conv2D(filters,
                                kernel_size = (3, 3),
                                strides = (1, 1),
                                activation = 'relu',
                                name = name + '_' + str(i))(out_tensor)

        # Add a final max pooling layer. We down scale by a factor of 2 x 2, and
        # move with a 2 x 2 stride.
        out_tensor = MaxPooling2D((2, 2), strides = (2, 2))(out_tensor)
        
        return out_tensor


    def fc_block(self, in_tensor, use_batchnorm, name):
        '''
            Adds a fully connected layer of specifically 4096 neurons to the
            model with a dropout of 0.5.

            Args:
                in_tensor (tensor):     Input tensor from previous layer

                use_batchnorm (bool):   Specify wether or not to include 
                                        batchnormalisation in the fully connec-
                                        ted layer.

                name (str):             The name of the fully connected layer

            Returns:
                Tensor with the fully connected layer
        '''

        # Add a fully connected layer with 4096 neurons. This is basically a
        # linear regression of the form output = activation(dot(input, kernel) +
        # bias), where where activation is the element-wise activation function
        # passed as the activation argument, kernel is a weights matrix created
        # by the layer, and bias is a bias vector created by the layer.
        # Activation is utilised (relu)
        out_tensor = Dense(4096, activation = 'relu', name = name)(in_tensor)

        # Add batchnormalisation to the layer if specified
        if use_batchnorm:
            out_tensor = BatchNormalization()(out_tensor)


        # We apply Hinton's dropout to the input with a rate of 0.5.
        # Dropout is a technique where randomly selected neurons are ignored
        # during training. They are "dropped-out randomly. This means that
        # their contribution to the activation of downstream neurons is
        # temporally removed on the forward pass and any weight updates are not
        # applied to the neuron on the backward pass.
        # Remember we train in batches. As such only 50% of the weights
        # (neurons) of this layer will be updated on each pass. This helps to
        # prevent over fitting.
        out_tensor = Dropout(0.5)(out_tensor)
        
        return out_tensor

    def create(self, size, use_batchnorm, include_top):
        '''
            Creates the VGG16 network architecture and loads the pretrained
            weights.

            Args:
                size (tuple):           Pixel size of input images. Default for 
                                        VGG is (224 x 224).

                use_batchnorm (bool):   Specify wether or not to use batchnorma-
                                        lisation.

                include_top (bool):     Specify wether or not to include the 
                                        last fully connected output layer. Will
                                        have to be specified manually if not.

            Returns:
                None
        '''

        # If we use another imagesize, the dense layers have to be retrained
        #if size != (224, 224):

        #    include_top = False

        # Define input layer (hard coded input shape, as VGG16 was trained on 
        # 224x224 images
        inputs = Input(shape = size + (3,))

        # Preprocess the images (subtract mean and flip channel). Lambda
        # basically wraps a function (in this case a transformation) as a layer
        # object.
        norm_layer = Lambda(self.vgg_preprocess,
                            output_shape = size + (3,),
                            name = 'norm_layer')(inputs)

        # Add the zero padded convolutional layers (13)
        conv_layer_1 = self.conv_block(norm_layer, 2, 64, name = 'conv_layer_1')
        conv_layer_2 = self.conv_block(conv_layer_1, 2, 128, name = 'conv_layer_2')
        conv_layer_3 = self.conv_block(conv_layer_2, 3, 256, name = 'conv_layer_3')
        conv_layer_4 = self.conv_block(conv_layer_3, 3, 512, name = 'conv_layer_4')
        conv_layer_5 = self.conv_block(conv_layer_4, 3, 512, name = 'conv_layer_5')

        # We then flatten the 3D output to a one dimentional vector for the
        # fully connected layer.
        flat_layer = Flatten(name = 'flat_layer')(conv_layer_5)

        # We then add two fully connected layers of 4096 neurons 
        if use_batchnorm:
            fc_layer_1 = self.fc_block(flat_layer,
                                       use_batchnorm = True,
                                       name = 'fc_layer_1')
            fc_layer_2 = self.fc_block(fc_layer_1,
                                       use_batchnorm = True, 
                                       name = 'fc_layer_2')

        else:
            fc_layer_1 = self.fc_block(flat_layer,
                                       use_batchnorm = False,
                                       name = 'fc_layer_1')
            fc_layer_2 = self.fc_block(fc_layer_1,
                                       use_batchnorm = False, 
                                       name = 'fc_layer_2')


        # Finally we add the output layer. ImageNet has 1000 categories, meaning
        # our output layer will need 1000 neurons. As standard, we use softmax
        # activation so as to get an output that sums to one (propability like).
        predictions = Dense(1000,
                            activation = 'softmax',
                            name = 'prediction')(fc_layer_2)
        
        # We then specify the output layer and the pretrained weights based on 
        # wether or not the denselayers are to be included
        if not include_top:
            fname = 'vgg16-no-top.h5'
            output_layer = conv_layer_5

        else:
            output_layer = predictions

            if use_batchnorm:
                fname = 'vgg16_bn.h5'
            else:
                fname = 'vgg16.h5'

        # Initialise model
        self.model = Model(inputs = inputs, outputs = output_layer)

        # Finally load the pretrained weights and cache them
        self.model.load_weights(os.path.join(os.path.dirname(__file__), self.WEIGHT_PATH + fname))
