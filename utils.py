import bcolz
import os
import numpy as np
import threading
import matplotlib.pyplot as plt

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

    
def plot_multi(im, dim = (4,4), figsize = (6,6), **kwargs ):
    
    plt.figure(figsize = figsize)
    
    for i, img in enumerate(im):
        plt.subplot(*((dim) + (i + 1,)))
        plt.imshow(img, **kwargs)
        plt.axis('off')
        
    plt.tight_layout()


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


class BcolzArrayIterator(object):
    """
        Returns an iterator object into Bcolz carray files
        Original version by Thiago Ramon Gonçalves Montoya
        Docs (and discovery) by @MPJansen
        Refactoring, performance improvements, fixes by Jeremy Howard j@fast.ai

        Example:
            X = bcolz.open('file_path/feature_file.bc', mode = 'r')
            y = bcolz.open('file_path/label_file.bc', mode = 'r')
            trn_batches = BcolzArrayIterator(X, y, batch_size = 64,
                                             shuffle = True)
            model.fit_generator(generator = trn_batches,
                                samples_per_epoch = trn_batches.N,
                                nb_epoch = 1)
        
        Args:
            X:          Input features
            y:          (optional) Input labels
            w:          (optional) Input feature weights
            batch_size: (optional) Batch size, defaults to 32
            shuffle:    (optional) Shuffle batches, defaults to false
            seed:       (optional) Provide a seed to shuffle, defaults to a random seed

        Returns: BcolzArrayIterator
    """

    def __init__(self, X, y = None, w = None, batch_size = 32, shuffle = False,
                 seed = None):

        if y is not None and len(X) != len(y):
            raise ValueError('X (features) and y (labels) should have the same length'
                             'Found: X.shape = %s, y.shape = %s' % (X.shape, y.shape))

        if w is not None and len(X) != len(w):
            raise ValueError('X (features) and w (weights) should have the same length'
                             'Found: X.shape = %s, w.shape = %s' % (X.shape, w.shape))

        if batch_size % X.chunklen != 0:
            raise ValueError('batch_size needs to be a multiple of X.chunklen')

        self.chunks_per_batch = batch_size // X.chunklen
        self.X = X
        self.y = y if y is not None else None
        self.w = w if w is not None else None
        self.N = X.shape[0]
        self.batch_size = batch_size
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.shuffle = shuffle
        self.seed = seed


    def reset(self):
        self.batch_index = 0


    def next(self):

        with self.lock:
            if self.batch_index == 0:
                if self.seed is not None:
                    np.random.seed(self.seed + self.total_batches_seen)

                self.index_array = (np.random.permutation(self.X.nchunks + 1) if self.shuffle
                    else np.arange(self.X.nchunks + 1))

            #batches_x = np.zeros((self.batch_size,)+self.X.shape[1:])
            batches_x, batches_y, batches_w = [],[],[]
            for i in range(self.chunks_per_batch):

                current_index = self.index_array[self.batch_index]

                if current_index == self.X.nchunks:
                    batches_x.append(self.X.leftover_array[:self.X.leftover_elements])
                    current_batch_size = self.X.leftover_elements

                else:
                    batches_x.append(self.X.chunks[current_index][:])
                    current_batch_size = self.X.chunklen

                self.batch_index += 1
                self.total_batches_seen += 1

                idx = current_index * self.X.chunklen

                if not self.y is None: batches_y.append(self.y[idx: idx + current_batch_size])
                if not self.w is None: batches_w.append(self.w[idx: idx + current_batch_size])

                if self.batch_index >= len(self.index_array):
                    self.batch_index = 0
                    break

            batch_x = np.concatenate(batches_x)
            if self.y is None:
                return batch_x

            batch_y = np.concatenate(batches_y)
            if self.w is None:
                return batch_x, batch_y

            batch_w = np.concatenate(batches_w)
            return batch_x, batch_y, batch_w


    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)
        