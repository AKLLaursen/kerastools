import bcolz
import os
import numpy as np

from keras.preprocessing import image
from keras.utils import to_categorical, convert_all_kernels_in_model
from keras import backend as K

import tensorflow as tf

def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config = cfg))

def do_clip(arr, mx):
    
    clipped = np.clip(arr, (1 - mx) / 1, mx)
    
    return clipped / clipped.sum(axis = 1)[:, np.newaxis]


def get_batches(dirname, gen = image.ImageDataGenerator(), shuffle = True,
                batch_size = 4, class_mode = 'categorical',
                target_size = (224, 224)):
    
    return gen.flow_from_directory(dirname, target_size = target_size,
                                   class_mode = class_mode,
                                   shuffle = shuffle, batch_size = batch_size)

def to_plot(img):
    return np.rollaxis(img, 0, 1).astype(np.uint8)

def plot(img):
    plt.imshow(to_plot(img))

def get_data(path, target_size = (224, 224)):
    batches = get_batches(path, shuffle = False, batch_size = 1, class_mode = None, target_size = target_size)
    
    return np.concatenate([batches.next() for i in range(batches.samples)])

def onehot_encode(x):
    return to_categorical(x)


def save_array(fname, arr):
    
    c = bcolz.carray(arr, rootdir = fname, mode = 'w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]

def split_at(model, layer_type):
    layers = model.layers
    layer_idx = [index for index,layer in enumerate(layers)
                 if type(layer) is layer_type][-1]
    return layers[:layer_idx+1], layers[layer_idx+1:]

def get_classes(path):
    
    batches = get_batches(path + 'train', shuffle = False, batch_size = 1)
    val_batches = get_batches(path + 'valid', shuffle = False, batch_size = 1)
    test_batches = get_batches(path + 'test', shuffle = False, batch_size = 1)
    
    return (val_batches.classes, batches.classes, onehot_encode(val_batches.classes),
            onehot_encode(batches.classes), val_batches.filenames, batches.filenames,
            test_batches.filenames)

class MixIterator(object):
    
    def __init__(self, iters):
        
        self.iters = iters
        self.n = sum([it.n for it in self.iters])
        self.batch_size = sum([it.batch_size for it in self.iters])
        self.steps_per_epoch = int(self.n / self.batch_size)

    def reset(self):
        for it in self.iters: it.reset()

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        
        nexts = [next(it) for it in self.iters]
        
        n0 = np.concatenate([n[0] for n in nexts])
        n1 = np.concatenate([n[1] for n in nexts])
        
        return (n0, n1)
