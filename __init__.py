# The Imagerecognition models can be used with both the TensorFlow and the 
# Theano backend. However, it is set up using TensorFlow. As such, if Theano is 
# to be used, the TensorFlow image ordering has to set explicitly.
# Theano: 'th' = 'channels_first'.
# TensorFlow: 'tf': = 'channels_last'.
from keras import backend as K
K.set_image_data_format('channels_last')