import json

from keras.models import Model
from keras.layers import Input, Activation, Dense, BatchNormalization
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.applications.resnet50 import identity_block, conv_block

from .vgg16 import vgg_preprocess

# The Resnet 50 model can be used with both the TensorFlow and the Theano 
# backend. However, it is set up using Theano. As such, if TensorFlow is to be 
# used, the Theano image ordering has to set explicitly.
# Theano: 'th' = 'channels_first'.
# TensorFlow: 'tf': = 'channels_last'.
from keras import backend as K
K.set_image_data_format('channels_first')

class Resnet50():
    """
        A Python implementation of the Resnet 50 Imagenet model
    """

    def __init__(self, size = (224,224), include_top = True):
        self.FILE_PATH = 'http://files.fast.ai/models/'
        self.create(size, include_top)
        self.get_classes()

    def create(self, size, include_top):

        # Define input layer (hard coded input shape, as Resnet 50 was trained 
        # on 224 x 224 images
        inputs = Input(shape = (3,) + size)

        # Define first layer stack of Resnet 50.

        # Preprocess data by subtracting mean from imagenet and rearraging 
        # channels
        x = Lambda(vgg_preprocess)(inputs)

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
            Modifies the original Resnet50 network architecture and updates self.classes for new training data.
            
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
            Predict the labels of a set of images using the Resnet50 model.

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


