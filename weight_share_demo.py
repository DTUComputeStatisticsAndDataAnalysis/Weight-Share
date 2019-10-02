# -*- coding: utf-8 -*-
"""
Author:         Jacob Søgaard Larsen (jasla@dtu.dk)

Last revision:  24-09-2019
    
"""
import numpy as np

from weight_share.data import get_chimiometrie_2019_data, get_SWRI_data, dataaugment
from weight_share.training import train_weight_share, train_cnn, train_transfer
from weight_share.build_graphs import restore_predictor,restore_cnn_predictor
from weight_share.misc import read_losses
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

#%%
X_2019,Y_2019,XTest_2019 = get_chimiometrie_2019_data()
X_SWRI_all,Y_SWRI_all = get_SWRI_data()

#%%
f,ax = plt.subplots(1,2,figsize=(10,4))
ax[0].plot(X_2019[:500].T)
ax[1].plot(X_SWRI_all.T)

ax[0].set_title("Chimiometrie 2019")
ax[0].set_ylabel("Absorption")
ax[1].set_title("IDRC 2002")
ax[1].set_ylabel("Absorption")
f.tight_layout()

#%%
scatter_level_train, scatter_level_val = 0.1,0.1
nAugmentTrain, nAugmentVal = 10,10

np.random.seed(1234)
idxTrain,idxVal,idxTest = np.split(np.random.choice(X_SWRI_all.shape[0],size=X_SWRI_all.shape[0],replace=False), [int(.5*X_SWRI_all.shape[0]), int(.75*X_SWRI_all.shape[0])])

XTrain_small = X_SWRI_all[idxTrain]
YTrain_small = Y_SWRI_all[idxTrain]
XVal_small = X_SWRI_all[idxVal]
YVal_small = Y_SWRI_all[idxVal]
XTest_small = X_SWRI_all[idxTest]
YTest_small = Y_SWRI_all[idxTest]

XTrain_2019,XVal_2019,YTrain_2019,YVal_2019 = train_test_split(X_2019,Y_2019,test_size=.3)
muX_small = XTrain_small.mean()
sigmaX_small = XTrain_small.std()
    
muX_2019 = XTrain_2019.mean()
sigmaX_2019 = XTrain_2019.std()

XTrain_small,YTrain_small = dataaugment(XTrain_small,YTrain_small,
                                      do_shuffle=True, 
                                      nAugment=nAugmentTrain,
                                      muX=muX_small,
                                      sigmaX=sigmaX_small,
                                      betashift=scatter_level_train, 
                                      slopeshift=scatter_level_train,
                                      multishift=scatter_level_train)
    
XTrain_2019,YTrain_2019 = dataaugment(XTrain_2019,YTrain_2019,
                                      do_shuffle=True, 
                                      nAugment=nAugmentTrain,
                                      muX=muX_2019,
                                      sigmaX=sigmaX_2019,
                                      betashift=scatter_level_train, 
                                      slopeshift=scatter_level_train,
                                      multishift=scatter_level_train)


XVal_small,YVal_small = dataaugment(XVal_small,YVal_small,
                                  do_shuffle=True,
                                  muX=muX_small,
                                  sigmaX=sigmaX_small,
                                  nAugment=nAugmentVal,
                                  betashift=scatter_level_val, 
                                  slopeshift=scatter_level_val,
                                  multishift=scatter_level_val)

XVal_2019,YVal_2019 = dataaugment(XVal_2019,YVal_2019,
                                  do_shuffle=True,
                                  muX=muX_2019,
                                  sigmaX=sigmaX_2019,
                                  nAugment=nAugmentVal,
                                  betashift=scatter_level_val, 
                                  slopeshift=scatter_level_val,
                                  multishift=scatter_level_val)

# as there are many zeros in the Chimiometrie 2019 references, we add noise to these
for i in range(3):
    idx = np.where(YTrain_2019[:,i]==0)[0]
    YTrain_2019[idx,i] = np.random.normal(0,scale=1e-2,size=idx.shape)

for i in range(3):
    idx = np.where(YVal_2019[:,i]==0)[0]
    YVal_2019[idx,i] = np.random.normal(0,scale=1e-2,size=idx.shape)
    
weights_2019 = [1/value.astype(np.float32) for value in YTrain_2019.mean(axis=0)]

XTrain_small = np.expand_dims(XTrain_small,axis=-1)
XVal_small = np.expand_dims(XVal_small,axis=-1)
XTrain_2019 = np.expand_dims(XTrain_2019,axis=-1)
XVal_2019 = np.expand_dims(XVal_2019,axis=-1)

#%%
f,ax = plt.subplots(1,2,figsize=(10,4))
ax[0].plot(XTrain_2019[:2000,:,0].T)
ax[1].plot(XTrain_small[:,:,0].T)

ax[0].set_title("Chimiometrie 2019")
ax[0].set_ylabel("Absorption")
ax[1].set_title("SWRI")
ax[1].set_ylabel("Absorption")
f.tight_layout()

#%% Train weight share 
save_path = "./saved_sessions/Weight_share"
#train_new_predictor = True
train_new_predictor = False
if train_new_predictor:
    predictor = train_weight_share(XTrain_small, YTrain_small, XTrain_2019,YTrain_2019,
                                   XVal_small, YVal_small, XVal_2019,YVal_2019,
                                   save_path=save_path,
                                   batch_size=128,
                                   keep_prob=[.95]*3, 
                                   n_updates=50_000,
                                   LOG_PERIOD=250,
                                   patience_factor=5,weights_2019=weights_2019, ema_decay=0.99, 
                                   padding="SAME",learning_rate=1e-3, name='CNN_weight_sharing', 
                                   name_small="SWRI",verbose=False)
else:
    predictor = restore_predictor(save_path, dims_small=XTrain_small.shape[1],dims_2019=XTrain_2019.shape[1],name="CNN_weight_sharing",
                                  net="SWRI",verbose=False,padding="SAME")

#%% Do transfer learning
save_path_2019 = "./saved_sessions/Chim_2019"
#save_path_transfer = "./saved_sessions/transfer_10000"
save_path_transfer = "./saved_sessions/transfer_5000"
#train_new_predictor_2019 = True
train_new_predictor_2019 = False
#train_net_predictor_transfer = True
train_net_predictor_transfer = False
if train_new_predictor_2019:
    train_cnn(XTrain_2019,YTrain_2019,XVal_2019,YVal_2019,
              save_path=save_path_2019,
              batch_size=128, 
              keep_prob=[.95]*3, 
              n_updates=50_000, 
              LOG_PERIOD=250,
#              n_updates=100, 
#              LOG_PERIOD=5,
              patience_factor=5, weights=weights_2019, ema_decay=0.99, 
              padding="SAME", learning_rate=1e-3, name='CNN_Chim_2019', 
              verbose=False,return_predictor=False)
    
elif train_net_predictor_transfer:
    predictor_transfer = train_transfer(XTrain_small,YTrain_small,XVal_small,YVal_small,
                                        restore_path=save_path_2019,
                                        save_path=save_path_transfer,
                                        batch_size=128,
                                        keep_prob=[.95]*3,
                                        n_updates=5_000,
                                        LOG_PERIOD=250,
#                                        n_updates=100,
#                                        LOG_PERIOD=5,
                                        patience_factor=50,
                                        ema_decay=0.99, 
                                        padding="SAME", learning_rate=1e-3, 
                                        name_large='CNN_Chim_2019', 
                                        name_small="SWRI",
                                        verbose=False,return_predictor=True)
else:
    predictor_transfer = restore_cnn_predictor(save_path_transfer, dims=XTrain_small.shape[1], name="CNN_SWRI_NIR",net="SWRI",
                                               verbose=False,padding="SAME",large_graph=False)
    
    
#%% Inspect losses
LOSSES_TRAIN_small,LOSSES_TRAIN_WS_2019,LOSSES_VAL = read_losses(save_path)
n_iter = len(LOSSES_TRAIN_small['CNN_weight_sharing/cost/cost']) 
iters = np.arange(250,n_iter*250+1,250)

f = plt.figure(figsize=(15,5))
ax1 = plt.subplot(131)
ax1.semilogy(iters,LOSSES_TRAIN_small['CNN_weight_sharing/cost/cost'],label="Train SWRI")
ax1.plot(iters,LOSSES_TRAIN_WS_2019['CNN_weight_sharing/cost/cost'],label="Train Chimio")
ax1.plot(iters,LOSSES_VAL['CNN_weight_sharing/cost/small/RMSE'],label="Val SWRI")
ax1.plot(iters,LOSSES_VAL['CNN_weight_sharing/cost/2019/WRMSE'],label="Val Chimio")
ax1.plot(iters,LOSSES_VAL['CNN_weight_sharing/cost/cost'],label="Val Combined")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Error")
ax1.set_title("Weight Share Training")
ax1.legend()
ax1.set_ylim(.3,10)

LOSSES_TRAIN_2019,LOSSES_VAL_2019 = read_losses(save_path_2019)
LOSSES_TRAIN_Transfer,LOSSES_VAL_Transfer= read_losses(save_path_transfer)
n_iter = len(LOSSES_TRAIN_2019['CNN_Chim_2019/cost/cost']) 
n_iter_transfer = len(LOSSES_TRAIN_Transfer['CNN_SWRI_NIR/cost/cost'])
iters = np.arange(250,n_iter*250+1,250)
iters_transfer = np.arange(250,250*n_iter_transfer+1,250)
ax2 = plt.subplot(132,sharey=ax1)
ax2.semilogy(iters,LOSSES_TRAIN_2019['CNN_Chim_2019/cost/cost'],label="Training")
ax2.plot(iters,LOSSES_VAL_2019['CNN_Chim_2019/cost/MT/cost'],label="Validation")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Error")
ax2.legend()
ax2.set_title("Pre-training")

ax3 = plt.subplot(133,sharey=ax1)
ax3.semilogy(iters_transfer,LOSSES_TRAIN_Transfer['CNN_SWRI_NIR/cost/cost'],label="Training")
ax3.plot(iters_transfer,LOSSES_VAL_Transfer['CNN_SWRI_NIR/cost/MT/cost'],label="Validation")
ax3.set_xlabel("Iteration")
ax3.legend()
ax3.set_title("Transfering")

f.tight_layout()
#%% Predict using Weight Share
preds_small_all =predictor.predict_small(X_SWRI_all)
preds_small_train = predictor.predict_small(X_SWRI_all[idxTrain])
preds_small_val = predictor.predict_small(X_SWRI_all[idxVal])
preds_small_test = predictor.predict_small(XTest_small)

res_small_train = preds_small_train-Y_SWRI_all[idxTrain]
res_small_val = preds_small_val-Y_SWRI_all[idxVal]
res_small_test = preds_small_test-YTest_small

rmse_small_train = (res_small_train**2).mean()**(1/2)
rmse_small_val = (res_small_val**2).mean()**(1/2)
rmse_small_test = (res_small_test**2).mean()**(1/2)

# Predict using transfer learning
preds_transfer_all =predictor_transfer.predict(X_SWRI_all)
preds_transfer_train = predictor_transfer.predict(X_SWRI_all[idxTrain])
preds_transfer_val = predictor_transfer.predict(X_SWRI_all[idxVal])
preds_transfer_test = predictor_transfer.predict(XTest_small)

res_transfer_train = preds_transfer_train-Y_SWRI_all[idxTrain]
res_transfer_val = preds_transfer_val-Y_SWRI_all[idxVal]
res_transfer_test = preds_transfer_test-YTest_small

rmse_transfer_train = (res_transfer_train**2).mean()**(1/2)
rmse_transfer_val = (res_transfer_val**2).mean()**(1/2)
rmse_transfer_test = (res_transfer_test**2).mean()**(1/2)

with plt.style.context("ggplot"):
    f,ax=plt.subplots(2,2,figsize=(15,10),sharex="row",sharey="row")
    ax[0,0].plot([YTest_small.min(),YTest_small.max()],[YTest_small.min(),YTest_small.max()],c="gray",ls="--")
    ax[0,0].scatter(Y_SWRI_all[idxTrain],preds_small_train,s=5,c="black",label="Train")
    ax[0,0].scatter(Y_SWRI_all[idxVal],preds_small_val,s=5,c="green",label="Val")
    ax[0,0].scatter(YTest_small,preds_small_test,s=5,c="red",label="Test")
    ax[0,0].legend()
    ax[0,0].set_xlabel("References (mass %)")
    ax[0,0].set_ylabel("Predictions (mass %)")
    ax[0,0].set_title("Weight Share")
    
    ax[1,0].axhline(0,c="gray",ls="--",zorder=0)
    ax[1,0].boxplot([res_small_train,res_small_val,res_small_test],zorder=1)
    ax[1,0].set_xticklabels(["Train","Validation","Test"],zorder=1)
    ax[1,0].set_ylabel(r"$\hat{y}-y$",zorder=1)
    
    print("Weight Share:")
    print(f"RMSE Train: {rmse_small_train:.3f}")
    print(f"RMSE Val: {rmse_small_val:.3f}")
    print(f"RMSE Test: {rmse_small_test:.3f}")
    print()
    
    
    
    ax[0,1].plot([YTest_small.min(),YTest_small.max()],[YTest_small.min(),YTest_small.max()],c="gray",ls="--")
    ax[0,1].scatter(Y_SWRI_all[idxTrain],preds_transfer_train,s=5,c="black",label="Train")
    ax[0,1].scatter(Y_SWRI_all[idxVal],preds_transfer_val,s=5,c="green",label="Val")
    ax[0,1].scatter(YTest_small,preds_transfer_test,s=5,c="red",label="Test")
    ax[0,1].legend()
    ax[0,1].set_xlabel("References (mass %)")
    ax[0,1].set_ylabel("Predictions (mass %)")
    ax[0,1].set_title("Transfer Learning")
    
    ax[1,1].axhline(0,c="gray",ls="--",zorder=0)
    ax[1,1].boxplot([res_transfer_train,res_transfer_val,res_transfer_test],zorder=1)
    ax[1,1].set_xticklabels(["Train","Validation","Test"],zorder=1)
    ax[1,1].set_ylabel(r"$\hat{y}-y$",zorder=1)
    
    f.tight_layout()
    print("Transfer Learning")
    print(f"RMSE Train: {rmse_transfer_train:.3f}")
    print(f"RMSE Val: {rmse_transfer_val:.3f}")
    print(f"RMSE Test: {rmse_transfer_test:.3f}")

