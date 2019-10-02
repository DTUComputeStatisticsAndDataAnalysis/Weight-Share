# -*- coding: utf-8 -*-
"""
Author:         Jacob Søgaard Larsen (jasla@dtu.dk)

Last revision:  24-09-2019
    
"""

import tensorflow as tf
import numpy as np

def conv1d(Input, kernel_length, n_outputs, name, strides=1, activation=True, use_bias=True, padding="SAME", reuse=False, uniform=True, alpha=0.1):
    with tf.variable_scope(name, reuse=reuse):
        n_filters_in = Input.get_shape().as_list()[-1]
        
        W = tf.get_variable(name='w_'+name,
                            shape=[kernel_length,n_filters_in,n_outputs],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=uniform))
        if use_bias:
            b = tf.get_variable(name='b_' + name,
                                shape=[n_outputs],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=uniform))
        
        if (activation is not None) and (activation is not False):
            out = tf.nn.conv1d(Input,W,stride=strides, padding=padding)
            if use_bias:
                out = tf.nn.bias_add(out, b)
            
            if (activation is True) or (activation == "relu"):
                out = tf.nn.relu(out,name='out_'+name)
            elif (activation == "lrelu"):
                out = tf.nn.leaky_relu(out,alpha=alpha,name='out_'+name)
            elif (activation == "elu"):                
                out = tf.nn.elu(out,name="out_"+name)
            else:
                raise NotImplementedError
            
        else:
            if use_bias:
                out = tf.nn.conv1d(Input,W,stride=strides, padding=padding)
                out = tf.nn.bias_add(out, b,name='out_'+name)
            else:
                out = tf.nn.conv1d(Input,W,stride=strides, padding=padding,name="out_"+name)
                
                
                
        return out
    
    
def fullyConnected(x, output_size, name, activation=True, use_bias=True, reuse=False, uniform=True, alpha=0.1):
    with tf.variable_scope(name, reuse=reuse):
        input_size = x.shape[1:]
        input_size = int(np.prod(input_size))
        
        x = tf.reshape(x, [-1, input_size])
        
        W = tf.get_variable(name='w_'+name,
                            shape=[input_size, output_size],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=uniform))
        if use_bias:
            b = tf.get_variable(name='b_'+name,
                                shape=[output_size],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=uniform))

        if (activation is not None) and (activation is not False):
            out = tf.matmul(x, W)
            if use_bias:
                out = tf.add(out, b)
            
            if (activation is True) or (activation == "relu"):
                out = tf.nn.relu(out,name='out_'+name)
            elif (activation == "lrelu"):
                out = tf.nn.leaky_relu(out,alpha=alpha,name='out_'+name)
            elif (activation == "elu"):                
                out = tf.nn.elu(out,name="out_"+name)
            else:
                raise NotImplementedError
            
        else:
            if use_bias:
                out = tf.add(tf.matmul(x, W), b,name='out_'+name)
            else:
                out = tf.matmul(x, W, name='out_'+name)
                
        return out    
    
def maxpool1d(x, name, kernel_length=2, strides=2):
    with tf.name_scope(name):
        out = tf.layers.max_pooling1d   (x,
                             pool_size=kernel_length, #size of window
                             strides=strides,
                             padding='SAME',name='out_'+name)
        return out
    
def spatialdropout1d(x,name,keep_rate):
    with tf.name_scope(name):
        out = tf.nn.dropout(x, 
                            keep_rate,
                            noise_shape=[tf.shape(x)[0], 1, x.shape[2]],
                            name="out_"+name)
        return out

def batch_normalization1d(x, name, is_training, axis=1, center=True, scale=True, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        out = tf.layers.batch_normalization(x,
                                            axis=axis,
                                            training=is_training,
                                            name=name,
                                            center=center,
                                            scale=scale)
        
        return out