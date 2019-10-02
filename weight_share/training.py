# -*- coding: utf-8 -*-
"""
Author:         Jacob Søgaard Larsen (jasla@dtu.dk)

Last revision:  24-09-2019
    
"""
import tensorflow as tf
import numpy as np
import pickle
from .misc import generate_labelled_data, get_available_gpus
from .build_graphs import build_training_graph, build_eval_graph, build_MT_graph, restore_graph, weight_share_predictor
from .build_graphs import build_training_graph_Chim_2019, build_eval_graph_Chim_2019, build_MT_graph_Chim_2019, restore_graph_cnn, cnn_predictor
from .build_graphs import build_training_graph_transfer, build_eval_graph_transfer
from checkmate import BestCheckpointSaver
import os
#%%
def train_weight_share(XTrain_small, YTrain_small, XTrain_2019,YTrain_2019,
                  XVal_small, YVal_small, XVal_2019,YVal_2019,
                  batch_size, keep_prob, n_updates, LOG_PERIOD, save_path=None,
                  patience_factor=3, weights_2019=[1.,1.,1.], ema_decay=0.99, 
                  padding="SAME", learning_rate=1e-3, name='CNN_weight_sharing', 
                  name_small="small",verbose=False):
    
#    small_name = name_small.replace("_","")
    small_name = "_"+name_small
    
    tf.reset_default_graph()
    
    x_pl_small = tf.placeholder(tf.float32, [None, XTrain_small.shape[1],1],name="x_pl_small")
    x_pl_2019 = tf.placeholder(tf.float32, [None, XTrain_2019.shape[1],1],name="x_pl_2019")
    y_pl_small = tf.placeholder(tf.float32, [None,1], name='y_small')
    y_pl_2019 = tf.placeholder(tf.float32, [None,3], name='y_2019')
    n_batches = int(max(XTrain_small.shape[0],XTrain_2019.shape[0]) / batch_size)
    
    learning_rate_monitor = {"best":np.inf, 
                             "best_it":0,
                             "patience":patience_factor * n_batches,
                             "min_lr":3e-5, 
                             "factor":0.5, 
                             "lr":learning_rate
                             }
    
    print("Running session")
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    sess = tf.Session(config=config)
    
    # Deploy generators on the CPU
    with tf.device("/CPU:0"):
        print("Build dataset")
        # Training data
        train_dat_small,train_label_small = generate_labelled_data(x_pl_small, y_pl_small, sess, XTrain_small, YTrain_small, batch_size)
        train_dat_2019,train_label_2019 = generate_labelled_data(x_pl_2019, y_pl_2019, sess, XTrain_2019, YTrain_2019, batch_size)
        
        # Validation data
        val_dat_small,val_label_small = generate_labelled_data(x_pl_small, y_pl_small, sess, XVal_small, YVal_small, XVal_small.shape[0])
        val_dat_2019,val_label_2019 = generate_labelled_data(x_pl_2019, y_pl_2019, sess, XVal_2019, YVal_2019, XVal_2019.shape[0]//2)

    # If any GPU's are available, the model will be trained on he first. Otherwise it will be trained on the CPU
    gpus = get_available_gpus()
    if len(gpus) > 0:
        device = [gpus[0]]
    else:
        device = ["/device:CPU:0"]
        
        
    # Deploy models on the chosen device
    with tf.device(device[0]):
        keep_prob_pl = tf.placeholder_with_default([1.0]*len(keep_prob), shape=(len(keep_prob)),name="keep_prob")
        weights_2019_pl = np.array(weights_2019).reshape(1,-1).astype(np.float32)
        learning_rate_pl = tf.placeholder_with_default(learning_rate,name="learning_rate",shape=())
        
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as var_scope:
            print()
            print("Building training graphs")
            losses_train_small, losses_train_2019, train_op_small, train_op_2019, global_step = build_training_graph(x_pl_small=train_dat_small, 
                                                                                                                     x_pl_2019=train_dat_2019,  
                                                                                                                     y_small=train_label_small,
                                                                                                                     y_2019=train_label_2019,
                                                                                                                     weights_2019=weights_2019_pl, 
                                                                                                                     keep_prob_pl=keep_prob_pl,
                                                                                                                     learning_rate=learning_rate_pl,
                                                                                                                     scope=var_scope,
                                                                                                                     name=name,
                                                                                                                     name_small=small_name,
                                                                                                                     verbose=verbose,
                                                                                                                     padding=padding,
                                                                                                                     )
            
            var_scope.reuse_variables()
            print("Building graphs for validation")
            losses_eval_val = build_eval_graph(x_pl_small=val_dat_small, 
                                               x_pl_2019=val_dat_2019,  
                                               y_small=val_label_small,
                                               y_2019=val_label_2019,
                                               weights_2019=weights_2019_pl, 
                                               keep_prob_pl=keep_prob_pl,
                                               learning_rate=learning_rate_pl,
                                               scope=var_scope,
                                               name=name,
                                               name_small=small_name,
                                               verbose=verbose,
                                               padding=padding,
                                               )

    if verbose:
        print("Initializing")
    sess.run(tf.global_variables_initializer())
    
    if save_path is None:
        saver = None
        best_ckpt_saver = None
        do_save = False
    else:        
        var_lst = [n for n in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name) if "ExponentialMovingAverage" in n.name]
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        saver = tf.train.Saver(var_list=var_lst)
        best_ckpt_saver = BestCheckpointSaver(
                    save_dir=save_path,
                    num_to_keep=2,
                    maximize=False,
                    saver=saver
                    )
        
        do_save = True
            
    
    if verbose:
        print()
        print("Trainable parameters:")
        var_names = [(n.name,n.shape.as_list()) for n in tf.trainable_variables()]
        for var_name in var_names:
            print(var_name)
    print() 

    print("Training")
    
    LOSSES_TRAIN_small = []
    LOSSES_TRAIN_2019 = []
    LOSSES_VAL = []
    
    act_losses_train_small = {key:0 for key,_ in losses_train_small.items()}
    act_losses_train_2019 = {key:0 for key,_ in losses_train_2019.items()}
    
    for it in range(n_updates):
        
        _, batch_losses_small, _ = sess.run(fetches = [train_op_small, losses_train_small, global_step],
                                           feed_dict={keep_prob_pl:keep_prob,
                                                      learning_rate_pl:learning_rate_monitor["lr"]})

        _, batch_losses_2019 = sess.run(fetches = [train_op_2019, losses_train_2019],
                                        feed_dict={keep_prob_pl:keep_prob,
                                                   learning_rate_pl:learning_rate_monitor["lr"]})
        
        for key,val in batch_losses_small.items():
            act_losses_train_small[key] += val/LOG_PERIOD
            
        for key,val in batch_losses_2019.items():
            act_losses_train_2019[key] += val/LOG_PERIOD
            
            
        if ((it+1) % LOG_PERIOD == 0):
            feed_dict_eval= {learning_rate_pl:learning_rate_monitor["lr"]}
            
            valuesVal = list(losses_eval_val.values())
            act_valuesVal = sess.run(valuesVal,feed_dict=feed_dict_eval)
            
            # Eval on validation data    
            act_values_dict_val = {}
            for key, _ in losses_eval_val.items():
                act_values_dict_val[key] = 0
                    
            for key, value in zip(act_values_dict_val.keys(), act_valuesVal):
                    act_values_dict_val[key] += value
                    
            val_cost = act_values_dict_val[name+"/cost/cost"]
            val_RMSE_small = act_values_dict_val[name+"/cost/small/RMSE"]
            val_RMSE_2019 = act_values_dict_val[name+"/cost/2019/RMSE/1"], act_values_dict_val[name+"/cost/2019/RMSE/2"], act_values_dict_val[name+"/cost/2019/RMSE/3"]
            val_WRMSE_2019 =  act_values_dict_val[name+"/cost/2019/WRMSE"]
            
            val_cost_MT = act_values_dict_val[name+"/cost/MT/cost"]
            val_RMSE_small_MT = act_values_dict_val[name+"/cost/MT/small/RMSE"]
            val_RMSE_2019_MT = act_values_dict_val[name+"/cost/MT/2019/RMSE/1"], act_values_dict_val[name+"/cost/MT/2019/RMSE/2"], act_values_dict_val[name+"/cost/MT/2019/RMSE/3"]
            val_WRMSE_2019_MT =  act_values_dict_val[name+"/cost/MT/2019/WRMSE"]
            
            avg_cost_small = act_losses_train_small[name+"/cost/cost"]
            avg_cost_2019 = act_losses_train_2019[name+"/cost/cost"]
            avg_regu = act_losses_train_2019[name+"/cost/regularization"]
            
            LR = learning_rate_monitor["lr"]
            print(f"Update {it+1:>7d} / {n_updates:<7d}\t Train RMSE {name_small}:\t{avg_cost_small:>7.3f} Train WRMSE 2019:\t{avg_cost_2019:>7.3f}\t Regu: {avg_regu:>7.3f} Learning Rate {LR:>.3e}")
            print(f"\tVal:    Cost: {val_cost:7.3f}\t {name_small} RMSE:\t\t{val_RMSE_small:>7.3f} 2019 WRMSE:\t\t{val_WRMSE_2019:>7.3f}\t RMSE: {val_RMSE_2019[0]:>7.3f} {val_RMSE_2019[1]:>7.3f} {val_RMSE_2019[2]:>7.3f}")
            print(f"\tVal MT: Cost: {val_cost_MT:7.3f}\t {name_small} RMSE:\t\t{val_RMSE_small_MT:>7.3f} 2019 WRMSE:\t\t{val_WRMSE_2019_MT:>7.3f}\t RMSE: {val_RMSE_2019_MT[0]:>7.3f} {val_RMSE_2019_MT[1]:>7.3f} {val_RMSE_2019_MT[2]:>7.3f}")
            print()
            LOSSES_TRAIN_small.append(act_losses_train_small)
            LOSSES_TRAIN_2019.append(act_losses_train_2019)
            LOSSES_VAL.append(act_values_dict_val)
            
            act_losses_train_small = {key:0 for key,_ in losses_train_small.items()}
            act_losses_train_2019 = {key:0 for key,_ in losses_train_2019.items()}
            
            if val_cost_MT < learning_rate_monitor["best"]:
                learning_rate_monitor["best"] = val_cost_MT
                learning_rate_monitor["best_it"] = it
            elif (it-learning_rate_monitor["best_it"]) > learning_rate_monitor["patience"]:
                lr = learning_rate_monitor["lr"]
                new_lr = max(lr*learning_rate_monitor["factor"],learning_rate_monitor["min_lr"])
                learning_rate_monitor["lr"] = new_lr
                learning_rate_monitor["best_it"] = it
            
            if do_save:
                best_ckpt_saver.handle(val_cost_MT, sess, global_step)
    
    if do_save:
        with open(os.path.join(save_path,"losses.pkl"),"wb") as f:
            pickle.dump([LOSSES_TRAIN_small,LOSSES_TRAIN_2019,LOSSES_VAL],f)
                    
    print('Optimization Finished')
    
    if do_save:
        sess.close()
        tf.reset_default_graph()
        x_pl_small = tf.placeholder(tf.float32, [None, XTrain_small.shape[1],1],name="x_pl_small")
        x_pl_2019 = tf.placeholder(tf.float32, [None, XTrain_2019.shape[1],1],name="x_pl_2019")
        sess,yhat_small,yhat_2019 = restore_graph(save_path,x_pl_small,x_pl_2019,name=name,net=small_name,
                                                  verbose=verbose, padding=padding)
    else:
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as var_scope:
            yhat_small,yhat_2019,*_ = build_MT_graph(x_pl_small=x_pl_small, x_pl_2019=x_pl_2019,
                                                     keep_prob_pl=[1.]*3, verbose=verbose,
                                                     padding=padding,name=name,
                                                     name_small=small_name,scope=var_scope)
    
    predictor = weight_share_predictor(yhat_small,x_pl_small,yhat_2019,x_pl_2019,sess)
    
    return predictor
#%%
def train_cnn(XTrain,YTrain,XVal,YVal,
              batch_size, keep_prob, n_updates, LOG_PERIOD, save_path=None,
              patience_factor=3, weights=[1.,1.,1.], ema_decay=0.99, 
              padding="SAME", learning_rate=1e-3, name='CNN_Chim_2019', 
              verbose=False,return_predictor=False):
    
    tf.reset_default_graph()
    
    x_pl = tf.placeholder(tf.float32, [None, XTrain.shape[1],1],name="x_pl")
    y_pl = tf.placeholder(tf.float32, [None,3], name='y_pl')
    n_batches = int(XTrain.shape[0] / batch_size)
    
    learning_rate_monitor = {"best":np.inf, 
                             "best_it":0,
                             "patience":patience_factor * n_batches,
                             "min_lr":3e-5, 
                             "factor":0.5, 
                             "lr":learning_rate
                             }
    
    print("Running session")
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    sess = tf.Session(config=config)
    
    # Deploy generators on the CPU
    with tf.device("/CPU:0"):
        print("Build dataset")
        # Training data
        train_dat,train_label= generate_labelled_data(x_pl, y_pl, sess, XTrain, YTrain, batch_size)
        
        # Validation data
        val_dat,val_label = generate_labelled_data(x_pl, y_pl, sess, XVal, YVal, XVal.shape[0]//2)

    # If any GPU's are available, the model will be trained on he first. Otherwise it will be trained on the CPU
    gpus = get_available_gpus()
    if len(gpus) > 0:
        device = [gpus[0]]
    else:
        device = ["/device:CPU:0"]
        
    # Deploy models on the chosen device
    with tf.device(device[0]):
        keep_prob_pl = tf.placeholder_with_default([1.0]*len(keep_prob), shape=(len(keep_prob)),name="keep_prob")
        weights_pl = np.array(weights).reshape(1,-1).astype(np.float32)
        learning_rate_pl = tf.placeholder_with_default(learning_rate,name="learning_rate",shape=())
        
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as var_scope:
            print()
            print("Building training graphs")
            losses_train, train_op, global_step = build_training_graph_Chim_2019(x_pl=train_dat,y_pl=train_label,
                                                                                 weights=weights_pl,keep_prob_pl=keep_prob_pl,
                                                                                 learning_rate=learning_rate_pl,
                                                                                 scope=var_scope,
                                                                                 name=name,
                                                                                 verbose=verbose,
                                                                                 padding=padding,
                                                                                 )
            
            var_scope.reuse_variables()
            print("Building graphs for validation")
            losses_eval_val = build_eval_graph_Chim_2019(x_pl=val_dat,  
                                               y_pl=val_label,
                                               weights=weights_pl, 
                                               keep_prob_pl=keep_prob_pl,
                                               learning_rate=learning_rate_pl,
                                               scope=var_scope,
                                               name=name,
                                               verbose=verbose,
                                               padding=padding,
                                               )

    if verbose:
        print("Initializing")
    sess.run(tf.global_variables_initializer())
    
    if save_path is None:
        saver = None
        best_ckpt_saver = None
        do_save = False
    else:        
        var_lst = [n for n in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name) if "ExponentialMovingAverage" in n.name]
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        saver = tf.train.Saver(var_list=var_lst)
        best_ckpt_saver = BestCheckpointSaver(
                    save_dir=save_path,
                    num_to_keep=2,
                    maximize=False,
                    saver=saver
                    )
        
        do_save = True
            
    if verbose:
        print()
        print("Trainable parameters:")
        var_names = [(n.name,n.shape.as_list()) for n in tf.trainable_variables()]
        for var_name in var_names:
            print(var_name)
    print() 

    print("Training")
    
    LOSSES_TRAIN = []
    LOSSES_VAL = []
    
    act_losses_train = {key:0 for key,_ in losses_train.items()}
    
    for it in range(n_updates):
        
        _, batch_losses,_ = sess.run(fetches = [train_op, losses_train,global_step],
                                     feed_dict={keep_prob_pl:keep_prob,
                                                learning_rate_pl:learning_rate_monitor["lr"]})
        
        for key,val in batch_losses.items():
            act_losses_train[key] += val/LOG_PERIOD
            
            
        if ((it+1) % LOG_PERIOD == 0):
            feed_dict_eval= {learning_rate_pl:learning_rate_monitor["lr"]}
            
            valuesVal = list(losses_eval_val.values())
            act_valuesVal = sess.run(valuesVal,feed_dict=feed_dict_eval)
            
            # Eval on validation data    
            act_values_dict_val = {}
            for key, _ in losses_eval_val.items():
                act_values_dict_val[key] = 0
                    
            for key, value in zip(act_values_dict_val.keys(), act_valuesVal):
                    act_values_dict_val[key] += value
                    
            val_cost = act_values_dict_val[name+"/cost/cost"]
            val_RMSE = act_values_dict_val[name+"/cost/2019/RMSE/1"], act_values_dict_val[name+"/cost/2019/RMSE/2"], act_values_dict_val[name+"/cost/2019/RMSE/3"]
            val_WRMSE =  act_values_dict_val[name+"/cost/2019/WRMSE"]
            
            val_cost_MT = act_values_dict_val[name+"/cost/MT/cost"]
            val_RMSE_MT = act_values_dict_val[name+"/cost/MT/2019/RMSE/1"], act_values_dict_val[name+"/cost/MT/2019/RMSE/2"], act_values_dict_val[name+"/cost/MT/2019/RMSE/3"]
            val_WRMSE_MT =  act_values_dict_val[name+"/cost/MT/2019/WRMSE"]
            
            avg_cost = act_losses_train[name+"/cost/cost"]
            avg_regu = act_losses_train[name+"/cost/regularization"]
            
            LR = learning_rate_monitor["lr"]
            print(f"Update {it+1:>7d} / {n_updates:<7d}\t Train Cost 2019:\t{avg_cost:>7.3f}\t Regu: {avg_regu:>7.3f} Learning Rate {LR:>.3e}")
            print(f"\tVal:    Cost: {val_cost:7.3f}\t WRMSE:\t\t{val_WRMSE:>7.3f}\t RMSE: {val_RMSE[0]:>7.3f} {val_RMSE[1]:>7.3f} {val_RMSE[2]:>7.3f}")
            print(f"\tVal MT: Cost: {val_cost_MT:7.3f}\t WRMSE:\t\t{val_WRMSE_MT:>7.3f}\t RMSE: {val_RMSE_MT[0]:>7.3f} {val_RMSE_MT[1]:>7.3f} {val_RMSE_MT[2]:>7.3f}")
            print()
            LOSSES_TRAIN.append(act_losses_train)
            LOSSES_VAL.append(act_values_dict_val)
            
            act_losses_train = {key:0 for key,_ in losses_train.items()}
            
            if val_cost_MT < learning_rate_monitor["best"]:
                learning_rate_monitor["best"] = val_cost_MT
                learning_rate_monitor["best_it"] = it
            elif (it-learning_rate_monitor["best_it"]) > learning_rate_monitor["patience"]:
                lr = learning_rate_monitor["lr"]
                new_lr = max(lr*learning_rate_monitor["factor"],learning_rate_monitor["min_lr"])
                learning_rate_monitor["lr"] = new_lr
                learning_rate_monitor["best_it"] = it
            
            if do_save:
                best_ckpt_saver.handle(val_cost_MT, sess, global_step)
    
    if do_save:
        with open(os.path.join(save_path,"losses.pkl"),"wb") as f:
            pickle.dump([LOSSES_TRAIN,LOSSES_VAL],f)
                    
    print('Optimization Finished')
    if return_predictor:
        if do_save:
            sess.close()
            tf.reset_default_graph()
            x_pl = tf.placeholder(tf.float32, [None, XTrain.shape[1],1],name="x_pl")
            sess,yhat = restore_graph_cnn(save_path,x_pl,name=name,
                                                      verbose=verbose, padding=padding)
        else:
            with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as var_scope:
                yhat,*_ = build_MT_graph_Chim_2019(x_pl=x_pl,
                                                         keep_prob_pl=[1.]*3, verbose=verbose,
                                                         padding=padding,name=name,
                                                         scope=var_scope)
        
        predictor = cnn_predictor(yhat,x_pl,sess)
        
        return predictor
#%%    
def train_transfer(XTrain,YTrain,XVal,YVal,
                   batch_size, keep_prob, n_updates, LOG_PERIOD,
                   restore_path,
                   save_path=None,
                   patience_factor=3, ema_decay=0.99, 
                   padding="SAME", learning_rate=1e-3, 
                   name_large='CNN_Chim_2019', 
                   name_small="small",
                   verbose=False,return_predictor=False):
    
    name = f"CNN_{name_small}_NIR"
    tf.reset_default_graph()
    
    x_pl = tf.placeholder(tf.float32, [None, XTrain.shape[1],1],name="x_pl")
    y_pl = tf.placeholder(tf.float32, [None,1], name='y_pl')
    n_batches = int(XTrain.shape[0] / batch_size)
    
    learning_rate_monitor = {"best":np.inf, 
                             "best_it":0,
                             "patience":patience_factor * n_batches,
                             "min_lr":3e-5, 
                             "factor":0.5, 
                             "lr":learning_rate
                             }
    
    print("Running session")
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    
    # Deploy generators on the CPU
    with tf.device("/CPU:0"):
        print("Build dataset")
        # Training data
        train_dat,train_label= generate_labelled_data(x_pl, y_pl, sess, XTrain, YTrain, batch_size)
        
        # Validation data
        val_dat,val_label = generate_labelled_data(x_pl, y_pl, sess, XVal, YVal, XVal.shape[0])

    # If any GPU's are available, the model will be trained on he first. Otherwise it will be trained on the CPU
    gpus = get_available_gpus()
    if len(gpus) > 0:
        device = [gpus[0]]
    else:
        device = ["/device:CPU:0"]
        
    # Deploy models on the chosen device
    with tf.device(device[0]):
        keep_prob_pl = tf.placeholder_with_default([1.0]*len(keep_prob), shape=(len(keep_prob)),name="keep_prob")
        learning_rate_pl = tf.placeholder_with_default(learning_rate,name="learning_rate",shape=())
        
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as var_scope:
            print()
            print("Building training graphs")
            losses_train, train_op, global_step = build_training_graph_transfer(x_pl=train_dat,y_pl=train_label,
                                                                                keep_prob_pl=keep_prob_pl,
                                                                                learning_rate=learning_rate_pl,
                                                                                restore_path=restore_path,
                                                                                sess=sess,
                                                                                scope=var_scope,
                                                                                net=name_small,
                                                                                verbose=verbose,
                                                                                padding=padding,
                                                                                )
            
            var_scope.reuse_variables()
            print("Building graphs for validation")
            losses_eval_val = build_eval_graph_transfer(x_pl=val_dat,  
                                                        y_pl=val_label,
                                                        keep_prob_pl=keep_prob_pl,
                                                        learning_rate=learning_rate_pl,
                                                        scope=var_scope,
                                                        net=name_small,
                                                        verbose=verbose,
                                                        padding=padding,
                                                        )

    if verbose:
        print("Initializing")
    sess.run(tf.global_variables_initializer())
    
    if save_path is None:
        saver = None
        best_ckpt_saver = None
        do_save = False
    else:        
        var_lst = [n for n in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name) if "ExponentialMovingAverage" in n.name]
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        saver = tf.train.Saver(var_list=var_lst)
        best_ckpt_saver = BestCheckpointSaver(
                    save_dir=save_path,
                    num_to_keep=2,
                    maximize=False,
                    saver=saver
                    )
        
        do_save = True
            
    if verbose:
        print()
        print("Trainable parameters:")
        var_names = [(n.name,n.shape.as_list()) for n in tf.trainable_variables()]
        for var_name in var_names:
            print(var_name)
    print() 

    print("Training")
    
    LOSSES_TRAIN = []
    LOSSES_VAL = []
    
    act_losses_train = {key:0 for key,_ in losses_train.items()}
    
    for it in range(n_updates):
        
        _, batch_losses,_ = sess.run(fetches = [train_op, losses_train,global_step],
                                   feed_dict={keep_prob_pl:keep_prob,
                                              learning_rate_pl:learning_rate_monitor["lr"]})
        
        for key,val in batch_losses.items():
            act_losses_train[key] += val/LOG_PERIOD
            
            
        if ((it+1) % LOG_PERIOD == 0):
            feed_dict_eval= {learning_rate_pl:learning_rate_monitor["lr"]}
            
            valuesVal = list(losses_eval_val.values())
            act_valuesVal = sess.run(valuesVal,feed_dict=feed_dict_eval)
            
            # Eval on validation data    
            act_values_dict_val = {}
            for key, _ in losses_eval_val.items():
                act_values_dict_val[key] = 0
                    
            for key, value in zip(act_values_dict_val.keys(), act_valuesVal):
                    act_values_dict_val[key] += value
                    
            val_cost = act_values_dict_val[name+"/cost/cost"]
            
            val_cost_MT = act_values_dict_val[name+"/cost/MT/cost"]
            
            avg_cost = act_losses_train[name+"/cost/cost"]
            
            LR = learning_rate_monitor["lr"]
            
            print(f"Update {it+1:>7d} / {n_updates:<7d}\t Train cost: {avg_cost:>7.3f} Val cost: {val_cost:>7.3f} Val MT Cost: {val_cost_MT:>7.3f} Learning Rate {LR:.3e}")
            
            LOSSES_TRAIN.append(act_losses_train)
            LOSSES_VAL.append(act_values_dict_val)
            
            act_losses_train = {key:0 for key,_ in losses_train.items()}
            
            if val_cost_MT < learning_rate_monitor["best"]:
                learning_rate_monitor["best"] = val_cost_MT
                learning_rate_monitor["best_it"] = it
            elif (it-learning_rate_monitor["best_it"]) > learning_rate_monitor["patience"]:
                lr = learning_rate_monitor["lr"]
                new_lr = max(lr*learning_rate_monitor["factor"],learning_rate_monitor["min_lr"])
                learning_rate_monitor["lr"] = new_lr
                learning_rate_monitor["best_it"] = it
            
            if do_save:
                best_ckpt_saver.handle(val_cost_MT, sess, global_step)
    
    if do_save:
        with open(os.path.join(save_path,"losses.pkl"),"wb") as f:
            pickle.dump([LOSSES_TRAIN,LOSSES_VAL],f)
                    
    print('Optimization Finished')
    if return_predictor:
        if do_save:
            sess.close()
            tf.reset_default_graph()
            x_pl = tf.placeholder(tf.float32, [None, XTrain.shape[1],1],name="x_pl")
            sess,yhat = restore_graph_cnn(save_path,x_pl,name=name,
                                                      verbose=verbose, padding=padding)
        else:
            with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as var_scope:
                yhat,*_ = build_MT_graph_Chim_2019(x_pl=x_pl,
                                                         keep_prob_pl=[1.]*3, verbose=verbose,
                                                         padding=padding,name=name,
                                                         scope=var_scope)
        
        predictor = cnn_predictor(yhat,x_pl,sess)
        
        return predictor