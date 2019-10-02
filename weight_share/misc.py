# -*- coding: utf-8 -*-
"""
Author:         Jacob Søgaard Larsen (jasla@dtu.dk)

Last revision:  24-09-2019
    
"""

import pickle
import os
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.data import Dataset

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def generate_labelled_data(x_pl, y_pl, sess, X, y, batch_size, buffer_size=10000):
    dataset = Dataset.from_tensor_slices((x_pl,y_pl))
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    sess.run(iterator.initializer, feed_dict={x_pl:X, y_pl:y})
    return iterator.get_next()


def RMSE(y,yhat):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y,yhat)),axis=0))

def WRMSE(y,yhat,weights):
    return tf.reduce_mean(weights * RMSE(y,yhat))


def read_losses(path):
    """
        Function for reading and merging training and validation losses from training using Weight Share
    """
    with open(os.path.join(path,"losses.pkl"),"rb") as f:
        files = pickle.load(f)
    out = []
    for file in files:
        d = {key:[] for key in file[0]}
        
        for counter,dictionary in enumerate(file):
            for key,val in dictionary.items():
                d[key].append(val)
        
        out.append(d)
    return out
