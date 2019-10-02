# -*- coding: utf-8 -*-
"""
Author:         Jacob Søgaard Larsen (jasla@dtu.dk)

Last revision:  24-09-2019
    
"""

import tensorflow as tf
from .layers import conv1d, fullyConnected, spatialdropout1d, maxpool1d, batch_normalization1d

def CNN_small(x,is_training_pl,keep_prob_pl, scope, padding="SAME",net="small", reuse=False, verbose=True, getter=None):
    if verbose:
        print(f"Creating {net}")
        
    with tf.variable_scope(scope, reuse=reuse, custom_getter=getter):                                       
        Input = conv1d(x, name="c1",kernel_length=11, n_outputs=8, strides=1, activation="relu", padding=padding, reuse=reuse)
        Input = maxpool1d(Input,name="p1")
        Input = batch_normalization1d(Input, name = "bn1"+net,is_training=is_training_pl, axis=-1, reuse=reuse)
        
        if verbose:
            print("c1: ",Input.shape)
            
        Input = conv1d(Input, name="c2", kernel_length=11, n_outputs=8, strides=1, activation="relu",  padding=padding, reuse=reuse)
        Input = maxpool1d(Input,name="p2")
        Input = spatialdropout1d(Input, name="do1", keep_rate=keep_prob_pl[0])
        Input = batch_normalization1d(Input, name = "bn2"+net,is_training=is_training_pl, axis=-1, reuse=reuse)
        
        if verbose:
            print("c2: ",Input.shape)
        
        Input = conv1d(Input, name="c3", kernel_length=8, n_outputs=16, strides=1, activation="relu",  padding=padding, reuse=reuse)
        Input = maxpool1d(Input,name="p3")
        Input = batch_normalization1d(Input, name = "bn3"+net, is_training=is_training_pl, axis=-1, reuse=reuse)
        
        if verbose:
            print("c3: ",Input.shape)
            
        Input = conv1d(Input, name="c4", kernel_length=8, n_outputs=16, strides=1, activation="relu",  padding=padding, reuse=reuse)
        Input = maxpool1d(Input,name="p4")
        Input = spatialdropout1d(Input, name="do2", keep_rate=keep_prob_pl[1])
        Input = batch_normalization1d(Input, name = "bn4"+net, is_training=is_training_pl, axis=-1, reuse=reuse)
        
        if verbose:
            print("c4: ",Input.shape)

        Input = conv1d(Input, name="c5", kernel_length=6, n_outputs=24, strides=1, activation="relu",  padding=padding, reuse=reuse)
        Input = maxpool1d(Input,name="p5")
        Input = batch_normalization1d(Input, name = "bn5"+net, is_training=is_training_pl, axis=-1, reuse=reuse)
        
        if verbose:
            print("c5: ",Input.shape)
        
        Input = conv1d(Input, name="c6", kernel_length=6, n_outputs=24, strides=1, activation="relu",  padding=padding, reuse=reuse)
        Input = maxpool1d(Input,name="p6")
        Input = spatialdropout1d(Input, name="do3", keep_rate=keep_prob_pl[2])
        
        Input = tf.layers.flatten(Input)
        
        if verbose:
            print("flatten: ",Input.shape)
        
        Input = batch_normalization1d(Input, name="bn6"+net, is_training=is_training_pl, axis=-1, reuse=reuse)    
        
        f1 = fullyConnected(Input, name="f1"+net, output_size=10, activation="relu",  reuse=reuse)
        if verbose:
            print("f1: ",f1.shape)
        
        Input = batch_normalization1d(f1,name = "bn7"+net,is_training=is_training_pl,axis=-1, reuse=reuse)
        
        yhat = fullyConnected(Input,name="y"+net,output_size=1, activation="relu", reuse=reuse)
        
    return yhat

def CNN_large(x,is_training_pl,keep_prob_pl, scope, padding="SAME",net="_Chim2019",reuse=False, verbose=True,getter=None):
    if verbose:
        print(f"Creating {net}")
    with tf.variable_scope(scope, reuse=reuse, custom_getter=getter):                                       
        Input = conv1d(x, name="c1",kernel_length=11, n_outputs=8, strides=1, activation="relu",  padding=padding, reuse=reuse)
        Input = maxpool1d(Input,name="p1")
        Input = batch_normalization1d(Input, name = "bn1"+net,is_training=is_training_pl, axis=-1, reuse=reuse)
        
        if verbose:
            print("c1: ",Input.shape)
            
        Input = conv1d(Input, name="c2",kernel_length=11, n_outputs=8, strides=1, activation="relu",  padding=padding, reuse=reuse)
        Input = maxpool1d(Input,name="p2")
        Input = spatialdropout1d(Input, name="do1", keep_rate=keep_prob_pl[0])
        Input = batch_normalization1d(Input, name = "bn2"+net,is_training=is_training_pl, axis=-1, reuse=reuse)
        
        if verbose:
            print("c2: ",Input.shape)
        
        Input = conv1d(Input, name="c3", kernel_length=8, n_outputs=16, strides = 1, activation="relu",  padding=padding, reuse=reuse)
        Input = maxpool1d(Input,name="p3")
        Input = batch_normalization1d(Input, name = "bn3"+net, is_training=is_training_pl, axis=-1, reuse=reuse)
        
        if verbose:
            print("c3: ",Input.shape)
            
        Input = conv1d(Input, name="c4", kernel_length=8, n_outputs=16, strides = 1, activation="relu",  padding=padding, reuse=reuse)
        Input = maxpool1d(Input,name="p4")
        Input = spatialdropout1d(Input, name="do2", keep_rate=keep_prob_pl[1])
        Input = batch_normalization1d(Input, name = "bn4"+net, is_training=is_training_pl, axis=-1, reuse=reuse)
        
        if verbose:
            print("c4: ",Input.shape)

        Input = conv1d(Input, name="c5", kernel_length=6, n_outputs=24, strides = 1, activation="relu",  padding=padding, reuse=reuse)
        Input = maxpool1d(Input,name="p5")
        Input = batch_normalization1d(Input, name = "bn5"+net, is_training=is_training_pl, axis=-1, reuse=reuse)
        
        if verbose:
            print("c5: ",Input.shape)
        
        Input = conv1d(Input, name="c6", kernel_length=6, n_outputs=24, strides = 1, activation="relu",  padding=padding, reuse=reuse)
        Input = maxpool1d(Input,name="p6")
        Input = spatialdropout1d(Input, name="do3", keep_rate=keep_prob_pl[2])
        
        Input = tf.layers.flatten(Input)
        
        if verbose:
            print("flatten: ",Input.shape)
        
        Input = batch_normalization1d(Input, name="bn6"+net, is_training=is_training_pl, axis=-1, reuse=reuse)    
        
        
        f1 = fullyConnected(Input, name="f1"+net, output_size=30, activation="relu",  reuse=reuse)
        if verbose:
            print("f1: ",f1.shape)
        
        bn5_1 = batch_normalization1d(f1, name="bn7_1"+net, is_training=is_training_pl, axis=-1, reuse=reuse)
        
        yhat = fullyConnected(bn5_1, name="y"+net, output_size=3, activation="relu", reuse=reuse)
        
        
    return yhat
