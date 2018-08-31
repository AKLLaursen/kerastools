import os

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Flatten
from tensorflow.python.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.python.keras.utils import get_file

from tensorflow.python.keras.backend import image_data_format

from kerastools.imagerecon import ImageRecognition

def get_source_inputs(tensor, layer=None, node_index=None):
    """Returns the list of input tensors necessary to compute `tensor`.
    Output will always be a list of tensors
    (potentially with 1 element).
    Arguments:
        tensor: The tensor to start from.
        layer: Origin layer of the tensor. Will be
            determined via tensor._keras_history if not provided.
        node_index: Origin node index of the tensor.
    Returns:
        List of input tensors.
    """
    if not hasattr(tensor, '_keras_history'):
        return tensor
   
    if layer is None or node_index:
        layer, node_index, _ = tensor._keras_history
    if not layer._inbound_nodes:
        return [tensor]
    else:
        node = layer._inbound_nodes[node_index]
        if not node.inbound_layers:
            # Reached an Input layer, stop recursion.
            return node.input_tensors
        else:
            source_tensors = []
            for i in range(len(node.inbound_layers)):
                x = node.input_tensors[i]
                layer = node.inbound_layers[i]
                node_index = node.node_indices[i]
                previous_sources = get_source_inputs(x, layer, node_index)
                # Avoid input redundancy.
                for x in previous_sources:
                    if x not in source_tensors:
                        source_tensors.append(x)
            return source_tensors


def _obtain_input_shape(input_shape,
                        default_size,
                        min_size,
                        data_format,
                        require_flatten,
                        weights=None):
    """Internal utility del's input shape.
    Arguments:
            input_shape: Either None (will return the default network input shape),
                    or a user-provided shape to be validated.
            default_size: Default input width/height for the model.
            min_size: Minimum input width/height accepted by the model.
            data_format: Image data format to use.
            require_flatten: Whether the model is expected to
                    be linked to a classifier via a Flatten layer.
            weights: One of `None` (random initialization)
                    or 'imagenet' (pre-training on ImageNet).
                    If weights='imagenet' input channels must be equal to 3.
    Returns:
            An integer shape tuple (may include None entries).
    Raises:
            ValueError: In case of invalid argument values.
    """
    if weights != 'imagenet' and input_shape and len(input_shape) == 3:
        if data_format == 'channels_first':
            if input_shape[0] not in {1, 3}:
                logging.warning('This model usually expects 1 or 3 input channels. '
                                'However, it was passed an input_shape with ' +
                                str(input_shape[0]) + ' input channels.')
            default_shape = (input_shape[0], default_size, default_size)
        else:
            if input_shape[-1] not in {1, 3}:
                logging.warning('This model usually expects 1 or 3 input channels. '
                                'However, it was passed an input_shape with ' +
                                str(input_shape[-1]) + ' input channels.')
            default_shape = (default_size, default_size, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_size, default_size)
        else:
            default_shape = (default_size, default_size, 3)
    if weights == 'imagenet' and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting`include_top=True` '
                                 'and loading `imagenet` weights, '
                                 '`input_shape` should be ' + str(default_shape) + '.')
        return default_shape
    if input_shape:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError('`input_shape` must be a tuple of three integers.')
                if input_shape[0] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[1] is not None and input_shape[1] < min_size) or
                        (input_shape[2] is not None and input_shape[2] < min_size)):
                    raise ValueError('Input size must be at least ' + str(min_size) +
                                     'x' + str(min_size) + '; got '
                                     '`input_shape=' + str(input_shape) + '`')
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError('`input_shape` must be a tuple of three integers.')
                if input_shape[-1] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[0] is not None and input_shape[0] < min_size) or
                        (input_shape[1] is not None and input_shape[1] < min_size)):
                    raise ValueError('Input size must be at least ' + str(min_size) +
                                     'x' + str(min_size) + '; got '
                                     '`input_shape=' + str(input_shape) + '`')
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == 'channels_first':
                input_shape = (3, None, None)
            else:
                input_shape = (None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape=' + str(input_shape) + '`')
    return input_shape

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
                                          data_format = image_data_format(),
                                          require_flatten = include_top)
        
        # Ensure correct shape if input tensor is provided.
        if input_tensor is None:
            img_input = Input(shape = input_shape)
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
            
            
            wname = self.TF_WEIGHTS if include_top else self.TF_WEIGHTS_NO_TOP
            weights_path = get_file(wname, self.PATH + wname, cache_subdir = 'models')
                
            self.model.load_weights(weights_path)
