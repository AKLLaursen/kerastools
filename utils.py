import bcolz
import os
import numpy as np

from keras.preprocessing import image
from keras.utils import to_categorical, convert_all_kernels_in_model
from keras import backend as K

import tensorflow as tf

def do_clip(arr, mx):
    
    clipped = np.clip(arr, (1 - mx) / 1, mx)
    
    return clipped / clipped.sum(axis = 1)[:, np.newaxis]


def get_batches(dirname, gen = image.ImageDataGenerator(), shuffle = True,
                batch_size = 4, class_mode = 'categorical',
                target_size = (224, 224)):
    
    return gen.flow_from_directory(dirname, target_size = target_size,
                                   class_mode = class_mode,
                                   shuffle = shuffle, batch_size = batch_size)


def onehot_encode(x):
    return to_categorical(x)


def save_array(fname, arr):
    
    c = bcolz.carray(arr, rootdir = fname, mode = 'w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]


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
        self.multi = type(iters) is list
        
        if self.multi:
            self.N = sum([it[0].N for it in self.iters])
        else:
            self.N = sum([it.N for it in self.iters])

    def reset(self):
        for it in self.iters: it.reset()

    def __iter__(self):
        return self

    def next(self, *args, **kwargs):
        
        if self.multi:
            
            nexts = [[next(it) for it in o] for o in self.iters]
            
            n0 = np.concatenate([n[0] for n in nexts])
            n1 = np.concatenate([n[1] for n in nexts])
            
            return (n0, n1)
        
        else:
            
            nexts = [next(it) for it in self.iters]
            
            n0 = np.concatenate([n[0] for n in nexts])
            n1 = np.concatenate([n[1] for n in nexts])
            
            return (n0, n1)
