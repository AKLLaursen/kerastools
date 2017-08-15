import numpy as np
import json

from keras.models import Model
from keras.layers import (Input, Dense, BatchNormalization, Flatten, Dropout,
    Lambda)
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
from keras.preprocessing import image


# The VGG16 model can be used with both the TensorFlow and the Theano backend.
# However, it is set up using Theano. As such, if TensorFlow is to be used, the
# Theano image ordering has to set explicitly.
# Theano: 'th' = 'channels_first'.
# TensorFlow: 'tf': = 'channels_last'.
from keras import backend as K
K.set_image_data_format('channels_first')


def vgg_preprocess(images):
    """
        The VGG model awas trained in 2014. The creators subtracted the average
        of each of the three (R, G, B) channels first, such that the data in
        each channel has mean zero. The mean is calculated from the original
        ImageNet training data and is hard codet as provided by the VGG
        researchers.
        Further, the model was trained using Caffe, which expected
        the channels to be in (B, G, R) order, whereas Python by default uses
        (R, G, B). Thus we also need to reverse the order.

        Args:
            images (ndarray):   A numpy array containing colored images

        Returns:
            images (ndarray);   A numpy array of the same dimension with the
                                mean of the imagenet training data subtracted
                                and the channels reversed.
    """

    # Hard coded mean as provided by VGG researchers
    vgg_mean = np.array([123.68, 116.779, 103.939], dtype = np.float32) \
                 .reshape((3, 1, 1))

    # Subtract the mean
    images = images - vgg_mean

    # Return images in reverse ordering of channels, i.e. (R, G, B)
    return images[:, ::-1]


class Vgg16():
    """
        A Python implementation of the VGG 16 Imagenet model
    """

    def __init__(self, size = (224, 224), use_batchnorm = False,
                 include_top = True):

        # Path to the VGG 16 image weights. Consider changing these to a local
        # page
        self.FILE_PATH = 'http://files.fast.ai/models/'
        self.create(size, use_batchnorm, include_top)
        self.get_classes()


    def conv_block(self, in_tensor, layers, filters, name):
        """
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
        """

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
        """
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
        """

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

    def create(self, size, use_batchnorm, include_top = True):
        """
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
        """

        # If we use another imagesize, the dense layers have to be retrained
        if size != (224, 224):

            include_top = False

        # Define input layer (hard coded input shape, as VGG16 was trained on 
        # 224x224 images
        inputs = Input(shape = (3,) + size)

        # Preprocess the images (subtract mean and flip channel). Lambda
        # basically wraps a function (in this case a transformation) as a layer
        # object.
        norm_layer = Lambda(vgg_preprocess,
                            output_shape = (3,) + size,
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
            fname = 'vgg16_bn_conv.h5'
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
        self.model.load_weights(get_file(fname,
                                    self.FILE_PATH + fname,
                                    cache_subdir = 'models'))


    def get_classes(self):
        """
            Downloads the ImageNet class index file and loads it to 
            self.get_classes unless it is already cached.

            Args:
                None

            Returns:
                None
        """

        # Get the ImageNet class indexes and cache them
        fname = 'imagenet_class_index.json'
        fpath = get_file(fname,
                         self.FILE_PATH + fname,
                         cache_subdir = 'models')

        # Open file and parse json
        with open(fpath) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]


    def get_batches(self, path, gen = image.ImageDataGenerator(),
                    shuffle = True, batch_size = 8, class_mode = 'categorical',
                    target_size = (224, 224)):
        """
            Takes the path to a directory, and generates batches of a
            ugmented/normalized data. Batches are yielded indefinitely, in an
            infinite loop. Basically a wrapper around 
            image.ImageDataGenerator().flow_from_directory()

            We utilise the default settings
            ImageDataGenerator(featurewise_center = False,
                               samplewise_center = False,
                               featurewise_std_normalization = False,
                               samplewise_std_normalization = False,
                               zca_whitening = False,
                               zca_epsilon = 1e-6,
                               rotation_range = 0.,
                               width_shift_range = 0.,
                               height_shift_range = 0.,
                               shear_range = 0.,
                               zoom_range = 0.,
                               channel_shift_range = 0.,
                               fill_mode = 'nearest',
                               cval = 0.,
                               horizontal_flip = False,
                               vertical_flip = False,
                               rescale = None,
                               preprocessing_function = None,
                               data_format = K.image_data_format())

            Args:
                path (str):         Path to directory with images to flow from.
                generator (fnc):    Initialised image.ImageDataGenerator()
                                    class with arguments, if other than default
                                    values are wanted.
                shuffle (bol):      Indicates whether or not the data should be
                                    shuffled. Default is true.
                batch_size (int):   Size of the batches of data. Default is 8
                class_mode (str):   one of "categorical", "binary", "sparse" or 
                                    None. Default: "categorical". Determines the 
                                    type of label arrays that are returned: 
                                    "categorical" will be 2D one-hot encoded 
                                    labels, "binary" will be 1D binary labels, 
                                    "sparse" will be 1D integer labels. If 
                                    None, no labels are returned (the generator 
                                    will only yield batches of image data, which 
                                    is useful to use model.predict_generator(), 
                                    model.evaluate_generator(), etc.).

            Returns:
                An initialised ImageDataGenerator().flow_from_directory() object
                ready with batches to be passed to training function.
        """

        # Note all data is rezised to 224 x 224 pixel images
        return gen.flow_from_directory(path,
                                       target_size = target_size,
                                       class_mode = class_mode,
                                       shuffle = shuffle,
                                       batch_size = batch_size)


    def compile(self, lr = 0.001):
        """
            Compiles the model for training.

            Args:
                lr (float): The learningrate for which lr >= 0.

            Returns:
                None
        """

        # Configure the model for training. We use the Adam optimiser presented
        # by https://arxiv.org/abs/1412.6980v8. For loss function we use 
        # categorical cross entropy described here 
        # https://en.wikipedia.org/wiki/Cross_entropy, and for evaluating the
        # training we use simple accuracy.
        self.model.compile(optimizer = Adam(lr = lr),
                           loss = 'categorical_crossentropy',
                           metrics = ['accuracy'])


    def fit_batch(self, batches, val_batches, epochs):
        """
            Fits the model on data yielded batch-by-batch for a given number of
            epochs, where the batches are generated from a get_batches object.

            Args:
                batches (obj):      A get_batches object with training batches.
                val_batches (obj):  A get_batches object with valuation batches.
                epochs (int):       Number of full cycles of the training data.

            Returns:
                None
        """

        # Fits the models across the batches
        self.model.fit_generator(batches,
                                 steps_per_epoch = np.int(batches.samples / batches.batch_size),
                                 epochs = epochs,
                                 validation_data = val_batches,
                                 validation_steps = np.int(val_batches.samples / val_batches.batch_size))
    
    
    def fit_data(self, train_data, train_labels, val_data, val_labels, epochs, batch_size):
        """
            Fits the model on an entire dataset for a given number of epochs.

            Args:
                train_data (ndarray):   A numpy array with training data.
                train_labels (ndarray): A numpy array with training labels.
                val_data (ndarray):     A numpy array with validation data.
                val_labels (ndarray):   A numpy array with validation labels.
                epochs (int):           Number of full cycles of the training data.
                batch_size (int):       Size each full batch of data to cycle trough.

            Returns:
                None
        """
        self.model.fit(x = train_data,
                       y = train_labels,
                       batch_size = batch_size,
                       epochs = epochs,
                       validation_data = (val_data, val_labels))
        
    
    def ft(self, num):
        """
            Replace the last layer of the model with a Dense (fully connected) layer of num neurons.
            Will also lock the weights of all layers except the new layer so that we only learn
            weights for the last layer in subsequent training.

            Args:
                num (int) : Number of neurons in the Dense layer
                
            Returns:
                None
        """
        
        model = self.model
        
        # Remove last layer
        model.layers.pop()
        
        # Set the remaining layers to not be trainable
        for layer in model.layers:
            layer.trainable = False
        
        # Define new dense layer
        inputs = model.layers[-1].output
        predictions = Dense(num, activation='softmax', name = 'predictions')(inputs)
        
        # Intialise new model
        self.model = Model(inputs = model.layers[0].output, outputs = predictions)
        self.compile()
        
    def finetune(self, batches):
        """
            Modifies the original VGG16 network architecture and updates self.classes for new training data.
            
            Args:
                batches : A keras.preprocessing.image.ImageDataGenerator object.
                          See definition for get_batches().
        """
        
        # Initialise new model
        self.ft(batches.num_class)
        
        # Get a list of all the class labels
        classes = list(iter(batches.class_indices))
        
        # batches.class_indices is a dict with the class name as key and an index as value
        # eg. {'cats': 0, 'dogs': 1}

        # sort the class labels by index according to batches.class_indices and update model.classes
        for c in batches.class_indices:
            classes[batches.class_indices[c]] = c
        self.classes = classes


    def predict(self, images, details = False):
        """
            Predict the labels of a set of images using the VGG16 model.

            Args:
                imgs (ndarray): An array of n images (size: n x width x 
                                height x channels).
            
            Returns:
                preds (np.array) :  Highest confidence value of the predictions
                                    for each image.
                idxs (np.ndarray):  Class index of the predictions with the 
                                    max confidence.
                classes (list):     Class labels of the predictions with the max 
                                    confidence.
        """

        # Predict the probability of each class for each image
        all_preds = self.model.predict(imgs)

        # For each image get the index of the class with max probability
        idxs = np.argmax(all_preds, axis = 1)

        # Get the values of the highest probability for each image
        preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]

        # Get the label of the class with the highest probability for each image
        classes = [self.classes[idx] for idx in idxs]
        
        return np.array(preds), idxs, classes
        

    def test(self, path, batches):
        """
            Predicts the classes using the trained model on data yielded batch-by-batch.

            Args:
                path (string):  Path to the target directory. If test data, it must be
                                placed in a subdirectory eg. /unknown
                batch_size (int): The number of images to be considered in each batch.
            
            Returns:
                test_batches, numpy array(s) of predictions for the test_batches.
    
        """
        test_batches = self.get_batches(path,
                                        shuffle = False,
                                        batch_size = batch_size,
                                        class_mode = None)
        
        return test_batches, self.model.predict_generator(test_batches, test_batches.nb_sample)


