# -*- coding: utf-8 -*-
"""
Author:         Jacob Søgaard Larsen (jasla@dtu.dk)

Last revision:  24-09-2019
    
"""

import numpy as np
import tensorflow as tf
from .cnn import CNN_small, CNN_large
from .misc import RMSE, WRMSE
from checkmate import get_best_checkpoint
#%%
def build_training_graph(x_pl_small, x_pl_2019, y_small, y_2019, scope, keep_prob_pl,
                         weights_2019, nAugment=1, ema_decay=0.99, learning_rate=1e-3, 
                         name='CNN_weight_sharing', name_small="small",
                         verbose=True,padding="SAME"):
    """
        Function for building the training graphs for Weight Sharing
    """
    with tf.variable_scope("step", reuse=tf.AUTO_REUSE):    
        global_step = tf.get_variable(
            name="global_step",
            shape=[],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
            trainable=False,
        )
    
    
    yhat_small = CNN_small(x_pl_small, is_training_pl=True, keep_prob_pl=keep_prob_pl,
                           verbose=verbose, padding=padding, net=name_small, scope=scope)
    
    yhat_2019 = CNN_large(x_pl_2019, is_training_pl=True, keep_prob_pl=keep_prob_pl,
                         verbose=verbose, padding=padding, scope=scope)
    
    with tf.name_scope("ema" + "/ema_variables"):
        original_trainable_vars = {
            tensor.op.name: tensor
            for tensor
            in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "CNN_weight_sharing")
        }
        
        ema = tf.train.ExponentialMovingAverage(ema_decay)
        update_op = ema.apply(original_trainable_vars.values())
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)
        
        if verbose:
            for key in original_trainable_vars.keys():
                print(key)
        
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    # Fetch weights in y_Chim_2019
    var = [v for v in tf.trainable_variables() if v.name == f'{name}/y_Chim2019/w_y_Chim2019:0'][0]

    # Define costs
    regu_cost = 0.1 * (tf.reduce_sum(tf.abs(tf.multiply(var[:,0],var[:,1]))) + tf.reduce_sum(tf.abs(tf.multiply(var[:,0],var[:,2]))) + tf.reduce_sum(tf.abs(tf.multiply(var[:,1],var[:,2]))))
    rmse_small = RMSE(y_small,yhat_small)[0]
    wrmse_2019 = WRMSE(y_2019,yhat_2019,weights_2019)
    
    cost_small = rmse_small
    cost_2019 = wrmse_2019 + regu_cost
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    with tf.name_scope('opt'):
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
            train_op_small = optimizer.minimize(cost_small,global_step=global_step)
            train_op_2019 = optimizer.minimize(cost_2019)
    
    losses_small = {}
    losses_small[name+"/cost/cost"] = cost_small
    losses_small[name+"/lambdas/lr"] = learning_rate
    
    losses_2019 = {}
    losses_2019[name+"/cost/cost"] = cost_2019
    losses_2019[name+"/cost/regularization"] = regu_cost
    losses_2019[name+"/lambdas/lr"] = learning_rate
    

    return losses_small, losses_2019, train_op_small, train_op_2019, global_step

#%%
def build_training_graph_Chim_2019(x_pl, y_pl, scope, keep_prob_pl,
                                    weights, nAugment=1, ema_decay=0.99, learning_rate=1e-3, 
                                    name='CNN_Chim_2019',verbose=True,padding="SAME"):
    """
        Function for building the training graphs for training individually on Chimiometrie 2019 data set
    """
    with tf.variable_scope("step", reuse=tf.AUTO_REUSE):    
        global_step = tf.get_variable(
            name="global_step",
            shape=[],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
            trainable=False,
        )
    
    
    yhat = CNN_large(x_pl, is_training_pl=True, keep_prob_pl=keep_prob_pl,
                         verbose=verbose, padding=padding, scope=scope)
    
    with tf.name_scope("ema" + "/ema_variables"):
        original_trainable_vars = {
            tensor.op.name: tensor
            for tensor
            in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "CNN_Chim_2019")
        }
        
        ema = tf.train.ExponentialMovingAverage(ema_decay)
        update_op = ema.apply(original_trainable_vars.values())
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)
        
        if verbose:
            for key in original_trainable_vars.keys():
                print(key)
        
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    # Fetch weights in y_Chim_2019
    var = [v for v in tf.trainable_variables() if v.name == f'{name}/y_Chim2019/w_y_Chim2019:0'][0]

    # Define costs
    regu_cost = 0.1 * (tf.reduce_sum(tf.abs(tf.multiply(var[:,0],var[:,1]))) + tf.reduce_sum(tf.abs(tf.multiply(var[:,0],var[:,2]))) + tf.reduce_sum(tf.abs(tf.multiply(var[:,1],var[:,2]))))
    wrmse = WRMSE(y_pl,yhat,weights)
    
    cost = wrmse + regu_cost
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    with tf.name_scope('opt'):
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
            train_op = optimizer.minimize(cost,global_step=global_step)
    
    losses = {}
    losses[name+"/cost/cost"] = cost
    losses[name+"/cost/regularization"] = regu_cost
    losses[name+"/lambdas/lr"] = learning_rate
    

    return losses, train_op, global_step

#%%
def build_training_graph_transfer(x_pl, y_pl, scope, keep_prob_pl,sess,restore_path,
                                  nAugment=1, ema_decay=0.99, learning_rate=1e-3, 
                                  net='small',name_large="CNN_Chim_2019",verbose=True,padding="SAME"):
    """
        Function for building the training graphs for transfer learning
    """
    name = f"CNN_{net}_NIR"
    name_small = f"_{net}"
    
    
    skip_layers=["_f1_","_y_","bn6","bn7","global_step"]
    
    with tf.variable_scope("step", reuse=tf.AUTO_REUSE):    
        global_step = tf.get_variable(
            name="global_step",
            shape=[],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
            trainable=False,
        )
    
    
    yhat = CNN_small(x_pl, is_training_pl=True, keep_prob_pl=keep_prob_pl,
                           verbose=verbose, padding=padding, net=name_small, scope=scope)
    
    
    variables_to_restore = {v.name.split(":")[0]: v for v in tf.get_collection(
                            tf.GraphKeys.GLOBAL_VARIABLES)}
    
    variables_not_init = {v.name.split(":")[0]: v for v in tf.get_collection(
                            tf.GraphKeys.GLOBAL_VARIABLES)}
    
    old_keys = list(variables_to_restore.keys())
    for old_key in old_keys:
        if any([skip_layer in old_key for skip_layer in skip_layers]):
            variables_to_restore.pop(old_key)
            variables_not_init.pop(old_key)
        else:
            new_key = "CNN_Chim_2019/" + old_key.replace(f"{net}_NIR","Chim_2019").replace(f"{net}","Chim2019") + "/ExponentialMovingAverage"
#            new_key = old_key.replace(f"{net}","Chim_2019").replace(f"{net}","Chim2019") + "/ExponentialMovingAverage"
            variables_to_restore[new_key] = variables_to_restore.pop(old_key)
    
    saver = tf.train.Saver(var_list=variables_to_restore)
    saver.restore(sess, get_best_checkpoint(restore_path, select_maximum_value=False))
    
    
    
    with tf.name_scope("ema" + "/ema_variables"):
        original_trainable_vars = {
            tensor.op.name: tensor
            for tensor
            in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name)
        }
        
        ema = tf.train.ExponentialMovingAverage(ema_decay)
        update_op = ema.apply(original_trainable_vars.values())
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)
        
        if verbose:
            for key in original_trainable_vars.keys():
                print(key)
        
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    # Define costs
    cost = RMSE(y_pl,yhat)[0]
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    with tf.name_scope('opt'):
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(cost, global_step=global_step)
    
    losses = {}
    losses[name+"/cost/cost"] = cost
    losses[name+"/lambdas/lr"] = learning_rate
    

    return losses, train_op, global_step


#%%
def build_eval_graph(x_pl_small, x_pl_2019, y_small, y_2019, scope, weights_2019,
                     keep_prob_pl=[1.]*3, nAugment=1, ema_decay=0.99, learning_rate=1e-3, 
                     name='CNN_weight_sharing',verbose=True, 
                     name_small="small", padding="SAME"):
    """
        Function for building the evaluation graphs for Weight Sharing
    """
    
    
    yhat_small = CNN_small(x_pl_small, is_training_pl=False, keep_prob_pl=keep_prob_pl,
                             verbose=verbose, padding=padding, net=name_small,
                             scope=scope)
        
    yhat_2019 = CNN_large(x_pl_2019, is_training_pl=False, keep_prob_pl=keep_prob_pl,
                             verbose=verbose, padding=padding, scope=scope)
    
    yhat_small_MT,yhat_2019_MT,var,var_MT = build_MT_graph(x_pl_small,x_pl_2019,keep_prob_pl,verbose,padding,name,name_small,scope)

    regu_cost = 0.1 * (tf.reduce_sum(tf.abs(tf.multiply(var[:,0],var[:,1]))) + tf.reduce_sum(tf.abs(tf.multiply(var[:,0],var[:,2]))) + tf.reduce_sum(tf.abs(tf.multiply(var[:,1],var[:,2]))))    
    regu_cost_MT = 0.1 * (tf.reduce_sum(tf.abs(tf.multiply(var_MT[:,0],var_MT[:,1]))) + tf.reduce_sum(tf.abs(tf.multiply(var_MT[:,0],var_MT[:,2]))) + tf.reduce_sum(tf.abs(tf.multiply(var_MT[:,1],var_MT[:,2]))))    
        
    
    rmse_small = RMSE(y_small,yhat_small)[0]
    rmse_2019 = RMSE(y_2019,yhat_2019)
    wrmse_2019 = WRMSE(y_2019,yhat_2019,weights_2019)

    rmse_small_MT = RMSE(y_small,yhat_small_MT)[0]
    rmse_2019_MT = RMSE(y_2019,yhat_2019_MT)
    wrmse_2019_MT = WRMSE(y_2019,yhat_2019_MT,weights_2019)
    
    
    cost = rmse_small + wrmse_2019 + regu_cost
    cost_MT = rmse_small_MT + wrmse_2019_MT + regu_cost_MT
    
    losses = {}
    losses[name+"/cost/cost"] = cost
    losses[name+"/cost/MT/cost"] = cost_MT
    losses[name+"/cost/regularization"] = regu_cost
    losses[name+"/cost/MT/regularization"] = regu_cost_MT
    
    losses[name+"/cost/small/RMSE"] = rmse_small
    
    losses[name+"/cost/2019/WRMSE"] = wrmse_2019
    losses[name+"/cost/2019/RMSE/1"] = rmse_2019[0]
    losses[name+"/cost/2019/RMSE/2"] = rmse_2019[1]
    losses[name+"/cost/2019/RMSE/3"] = rmse_2019[2]
    
    losses[name+"/cost/MT/small/RMSE"] = rmse_small_MT
    
    losses[name+"/cost/MT/2019/WRMSE"] = wrmse_2019_MT
    losses[name+"/cost/MT/2019/RMSE/1"] = rmse_2019_MT[0]
    losses[name+"/cost/MT/2019/RMSE/2"] = rmse_2019_MT[1]
    losses[name+"/cost/MT/2019/RMSE/3"] = rmse_2019_MT[2]
    
    losses[name+"/lambdas/lr"] = learning_rate

    return losses

#%%
def build_eval_graph_Chim_2019(x_pl, y_pl, scope, weights,
                     keep_prob_pl=[1.]*3, nAugment=1, ema_decay=0.99, learning_rate=1e-3, 
                     name='CNN_Chim_2019',verbose=True, padding="SAME"):
    """
        Function for building the evaluation graphs for training individually on Chimiometrie 2019 data set
    """
    
    
    yhat = CNN_large(x_pl, is_training_pl=False, keep_prob_pl=keep_prob_pl,
                             verbose=verbose, padding=padding, scope=scope)
    
    yhat_MT,var,var_MT = build_MT_graph_Chim_2019(x_pl,keep_prob_pl,verbose,padding,name,scope)

    regu_cost = 0.1 * (tf.reduce_sum(tf.abs(tf.multiply(var[:,0],var[:,1]))) + tf.reduce_sum(tf.abs(tf.multiply(var[:,0],var[:,2]))) + tf.reduce_sum(tf.abs(tf.multiply(var[:,1],var[:,2]))))    
    regu_cost_MT = 0.1 * (tf.reduce_sum(tf.abs(tf.multiply(var_MT[:,0],var_MT[:,1]))) + tf.reduce_sum(tf.abs(tf.multiply(var_MT[:,0],var_MT[:,2]))) + tf.reduce_sum(tf.abs(tf.multiply(var_MT[:,1],var_MT[:,2]))))    
        
    
    rmse_2019 = RMSE(y_pl,yhat)
    wrmse_2019 = WRMSE(y_pl,yhat,weights)

    rmse_2019_MT = RMSE(y_pl,yhat_MT)
    wrmse_2019_MT = WRMSE(y_pl,yhat_MT,weights)
    
    
    cost = wrmse_2019 + regu_cost
    cost_MT = wrmse_2019_MT + regu_cost_MT
    
    losses = {}
    losses[name+"/cost/cost"] = cost
    losses[name+"/cost/MT/cost"] = cost_MT
    losses[name+"/cost/regularization"] = regu_cost
    losses[name+"/cost/MT/regularization"] = regu_cost_MT
    
    losses[name+"/cost/2019/WRMSE"] = wrmse_2019
    losses[name+"/cost/2019/RMSE/1"] = rmse_2019[0]
    losses[name+"/cost/2019/RMSE/2"] = rmse_2019[1]
    losses[name+"/cost/2019/RMSE/3"] = rmse_2019[2]
    
    losses[name+"/cost/MT/2019/WRMSE"] = wrmse_2019_MT
    losses[name+"/cost/MT/2019/RMSE/1"] = rmse_2019_MT[0]
    losses[name+"/cost/MT/2019/RMSE/2"] = rmse_2019_MT[1]
    losses[name+"/cost/MT/2019/RMSE/3"] = rmse_2019_MT[2]
    
    losses[name+"/lambdas/lr"] = learning_rate

    return losses

#%%
def build_eval_graph_transfer(x_pl, y_pl, scope,
                             keep_prob_pl=[1.]*3, nAugment=1, ema_decay=0.99, learning_rate=1e-3, 
                             net='small',verbose=True, padding="SAME"):
    """
        Function for building the evaluation graphs for transfer learning
    """
    name = f"CNN_{net}_NIR"
    name_small = f"_{net}"
    
    
    yhat = CNN_small(x_pl, is_training_pl=False, keep_prob_pl=keep_prob_pl,
                           verbose=verbose, padding=padding, net=name_small, scope=scope)
    
    yhat_MT = build_MT_graph_small(x_pl,keep_prob_pl,verbose,padding,net,scope)
        
    
    cost = RMSE(y_pl,yhat)[0]
    cost_MT = RMSE(y_pl,yhat_MT)[0]
    
    losses = {}
    losses[name+"/cost/cost"] = cost
    losses[name+"/cost/MT/cost"] = cost_MT
    
    losses[name+"/lambdas/lr"] = learning_rate

    return losses


#%%
def restore_graph(path,x_pl_small,x_pl_2019,name="CNN_weight_sharing",net="small",
                  verbose=False, padding="SAME"):
    
    """
        Function for restoring the Weight Sharing graphs from a checkpoint
    """
    
    net_name = f"_{net}"
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
        yhat_small = CNN_small(x_pl_small, is_training_pl=False,net=net_name,
                               keep_prob_pl=[1.]*3,verbose=verbose, padding=padding, 
                               scope=scope)
        
        yhat_2019 = CNN_large(x_pl_2019, is_training_pl=False, keep_prob_pl=[1.]*3,
                              verbose=verbose, padding=padding, scope=scope)
    
    
    ema = tf.train.ExponentialMovingAverage(decay=1)
    vars_to_restore = ema.variables_to_restore(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,name))
    
    old_keys = list(vars_to_restore.keys())
    for old_key in old_keys:
#        break
        if not name in old_key:
            vars_to_restore.pop(old_key)
        else:
            if "ExponentialMovingAverage" in old_key:
                new_key = name+"/"+old_key
            else:
                new_key = name+"/"+old_key+"/ExponentialMovingAverage"
            vars_to_restore[new_key] = vars_to_restore.pop(old_key)    
#        
    saver = tf.train.Saver(vars_to_restore)
    sess = tf.Session()
    saver.restore(sess, get_best_checkpoint(path, select_maximum_value=False))
    
    return sess,yhat_small,yhat_2019

#%%
def restore_graph_cnn(path,x_pl,name="CNN_Chim_2019", net=None,
                  verbose=False, padding="SAME",large_graph=True):
    
    """
        Function for restoring the Chimiometrie 2019 graph from a checkpoint
    """
    
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
        if large_graph:
            yhat = CNN_large(x_pl, is_training_pl=False, keep_prob_pl=[1.]*3,
                                  verbose=verbose, padding=padding, scope=scope)
        else:
            name_small = "_"+net
            yhat = CNN_small(x_pl, is_training_pl=False, keep_prob_pl=[1.]*3,
                             verbose=verbose, padding=padding, net=name_small,
                             scope=scope)
    
    
    ema = tf.train.ExponentialMovingAverage(decay=1)
    vars_to_restore = ema.variables_to_restore(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,name))
    
    old_keys = list(vars_to_restore.keys())
    for old_key in old_keys:
        if not name in old_key:
            vars_to_restore.pop(old_key)
        else:
            if "ExponentialMovingAverage" in old_key:
                new_key = name+"/"+old_key
            else:
                new_key = name+"/"+old_key+"/ExponentialMovingAverage"
            vars_to_restore[new_key] = vars_to_restore.pop(old_key)    
#        
    saver = tf.train.Saver(vars_to_restore)
    sess = tf.Session()
    saver.restore(sess, get_best_checkpoint(path, select_maximum_value=False))
    
    return sess,yhat

#%%
def build_MT_graph(x_pl_small,x_pl_2019,keep_prob_pl,verbose,padding,name,name_small,scope):
    """
        Function for building Mean Teacher (exponentially weighted parameters) graphs used in Weight Sharing training.
    """
    
    original_trainable_vars = {
        tensor.op.name.replace("/ExponentialMovingAverage",""): tensor
        for tensor
        in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"CNN_weight_sharing/") if "ExponentialMovingAverage" in tensor.op.name
    }
    
    
    if verbose:
        for key in original_trainable_vars.keys():
            print(key)
        
        
    def use_ema_variables(getter, name, *_, **__):
        assert scope.name+"/"+name in original_trainable_vars, "Unknown variable {}.".format(name)
        return original_trainable_vars[scope.name+"/"+name]
    
    with tf.variable_scope("Teacher_graph", reuse=tf.AUTO_REUSE):
    
#    print("Building validation graphs")
        yhat_small_MT = CNN_small(x_pl_small, is_training_pl=False, keep_prob_pl=keep_prob_pl,
                             verbose=verbose, padding=padding, getter=use_ema_variables,
                             net=name_small, scope=scope)
        
        yhat_2019_MT = CNN_large(x_pl_2019,is_training_pl=False, keep_prob_pl=keep_prob_pl,
                             verbose=verbose, padding=padding, getter=use_ema_variables,
                             scope=scope)
        
    var = [v for v in tf.trainable_variables() if v.name == f'{name}/y_Chim2019/w_y_Chim2019:0'][0]
    var_MT = [value for key,value in original_trainable_vars.items() if "y_Chim2019/w_y_Chim2019" in key][0]
    
    return yhat_small_MT,yhat_2019_MT,var,var_MT

#%%
def build_MT_graph_Chim_2019(x_pl_2019,keep_prob_pl,verbose,padding,name,scope):
    """
        Function for building Mean Teacher (exponentially weighted parameters) graph for the Chimiometrie 2019 data set.
    """
    
    original_trainable_vars = {
        tensor.op.name.replace("/ExponentialMovingAverage",""): tensor
        for tensor
        in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"CNN_Chim_2019/") if "ExponentialMovingAverage" in tensor.op.name
    }
    
    
    if verbose:
        for key in original_trainable_vars.keys():
            print(key)
        
        
    def use_ema_variables(getter, name, *_, **__):
        assert scope.name+"/"+name in original_trainable_vars, "Unknown variable {}.".format(name)
        return original_trainable_vars[scope.name+"/"+name]
    
    with tf.variable_scope("Teacher_graph", reuse=tf.AUTO_REUSE):
    
#    print("Building validation graphs")
        yhat_2019_MT = CNN_large(x_pl_2019,is_training_pl=False, keep_prob_pl=keep_prob_pl,
                             verbose=verbose, padding=padding, getter=use_ema_variables,
                             scope=scope)
        
    var = [v for v in tf.trainable_variables() if v.name == f'{name}/y_Chim2019/w_y_Chim2019:0'][0]
    var_MT = [value for key,value in original_trainable_vars.items() if "y_Chim2019/w_y_Chim2019" in key][0]
    
    return yhat_2019_MT,var,var_MT

#%%
def build_MT_graph_small(x_pl_small,keep_prob_pl,verbose,padding,net,scope):
    """
        Function for building Mean Teacher (exponentially weighted parameters)
    """
    name = f"CNN_{net}_NIR"
    name_small = "_"+net
    
    original_trainable_vars = {
        tensor.op.name.replace("/ExponentialMovingAverage",""): tensor
        for tensor
        in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,name) if "ExponentialMovingAverage" in tensor.op.name
    }
    
    
    if verbose:
        for key in original_trainable_vars.keys():
            print(key)
        
        
    def use_ema_variables(getter, name, *_, **__):
        assert scope.name+"/"+name in original_trainable_vars, "Unknown variable {}.".format(name)
        return original_trainable_vars[scope.name+"/"+name]
    
    with tf.variable_scope("Teacher_graph", reuse=tf.AUTO_REUSE):
    
#    print("Building validation graphs")
        yhat_small_MT = CNN_small(x_pl_small, is_training_pl=False, keep_prob_pl=keep_prob_pl,
                                  verbose=verbose, padding=padding, getter=use_ema_variables,
                                  net=name_small, scope=scope)
        
        
    return yhat_small_MT


#%%
class weight_share_predictor:
    """
        Class for performing predictions with deep CNN's trained using Weight Sharing
    """
    def __init__(self,yhat_small,x_pl_small,yhat_2019,x_pl_2019,sess):
        self.yhat_small = yhat_small
        self.x_pl_small = x_pl_small
        self.yhat_2019 = yhat_2019
        self.x_pl_2019 = x_pl_2019
        self.sess = sess
        self.dims_small = x_pl_small.get_shape().as_list()
        self.dims_2019 = x_pl_2019.get_shape().as_list()
        
    def predict_small(self,X):
        
        if len(X.shape) == 2:
            X = np.expand_dims(X,axis=-1)
        elif len(X.shape) != 3:
            raise Exception()
        
        if not X.shape[1] == self.dims_small[1]:
            raise Exception(f"Input dimension should be [None, {self.dims_small[1]}]")
        
        return self.yhat_small.eval({self.x_pl_small:X},session=self.sess)
    
    def predict_2019(self,X):
        if len(X.shape) == 2:
            X = np.expand_dims(X,axis=-1)
        elif len(X.shape) != 3:
            raise Exception()
        
        if not X.shape[1] == self.dims_2019[1]:
            raise Exception(f"Input dimension should be [None, {self.dims_2019[1]}]")
        
        return self.yhat_2019.eval({self.x_pl_2019:X},session=self.sess)


#%%
class cnn_predictor:
    """
        Class for performing predictions with neural networks
    """
    def __init__(self,yhat,x_pl,sess):
        self.yhat = yhat
        self.x_pl = x_pl
        self.sess = sess
        self.dims = x_pl.get_shape().as_list()
        
    def predict(self,X):
        if len(X.shape) == 2:
            X = np.expand_dims(X,axis=-1)
        elif len(X.shape) != 3:
            raise Exception()
        
        if not X.shape[1] == self.dims[1]:
            raise Exception(f"Input dimension should be [None, {self.dims[1]}]")
        
        return self.yhat.eval({self.x_pl:X},session=self.sess)

#%%
def restore_predictor(path, dims_small=650,dims_2019=550,name="CNN_weight_sharing",
                      net="small",verbose=False,padding="SAME"):
    
    """
        Function for restoring deep CNN's trained using Weight Share into a weight_share_predictor object
    """
    
#    tf.reset_default_graph()    
    x_pl_small = tf.placeholder(tf.float32, [None, dims_small,1],name="x_pl_small")
    x_pl_2019 = tf.placeholder(tf.float32, [None, dims_2019,1],name="x_pl_2019")
    sess,yhat_small,yhat_2019 = restore_graph(path,x_pl_small,x_pl_2019,name=name,net=net,
                                              verbose=verbose,padding=padding)
    
    predictor = weight_share_predictor(yhat_small,x_pl_small,yhat_2019,x_pl_2019,sess)
    
    return predictor
    
#%%
def restore_cnn_predictor(path, dims=550,name="CNN_Chim_2019",net=None,
                      verbose=False,padding="SAME",large_graph=True):
    
    """
        Function for restoring deep CNN's into a cnn_predictor object
    """
    
#    tf.reset_default_graph()    
    x_pl = tf.placeholder(tf.float32, [None, dims,1],name="x_pl")
    sess,yhat = restore_graph_cnn(path,x_pl,name=name,net=net,large_graph=large_graph,
                                  verbose=verbose,padding=padding)
    
    predictor = cnn_predictor(yhat,x_pl,sess)
    
    return predictor

