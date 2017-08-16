from __future__ import division, print_function

# The Imagerecognition models can be used with both the TensorFlow and the 
# Theano backend. However, it is set up using Theano. As such, if TensorFlow is 
# to be used, the Theano image ordering has to set explicitly.
# Theano: 'th' = 'channels_first'.
# TensorFlow: 'tf': = 'channels_last'.
from keras import backend as K
K.set_image_data_format('channels_first')