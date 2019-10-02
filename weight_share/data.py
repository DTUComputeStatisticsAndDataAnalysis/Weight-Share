# -*- coding: utf-8 -*-
"""
Author:         Jacob Søgaard Larsen (jasla@dtu.dk)

Last revision:  24-09-2019
    
"""
import requests
import pandas as pd
from io import BytesIO,StringIO
from zipfile import ZipFile
import numpy as np
from sklearn.utils import shuffle

def get_chimiometrie_2019_data():
    """
        Helper function to collect the Chimiometrie 2019 data set
        The data set consist of 
            6915 Training sample pairs (x,y)
        
        The response, y, has three columns:
            1: Soy oil
            2: Lucerne
            3: Barley
                
        The data set comes with 600 test samples, but without reference values for these.
            
        The function takes no input arguments.
        The function returns the tuple X,Y,XTest
            X:          Training samples (spectra)              (6915 x 550)
            Y:          Reference values for each sample        (6915 x 3)
            XTest:      Test samples (spectra)                  (600 x 550)

    """
    url_2019 = "https://chemom2019.sciencesconf.org/data/challenge_chimiometrie2019.zip"
    response_2019 = requests.get(url_2019)
    zipfile_2019 = ZipFile(BytesIO(response_2019.content))
        
    chunk_size = 10
    
    df_cal = pd.DataFrame()
    s = ""
    for counter,line in enumerate(zipfile_2019.open("XCAL.CSV").readlines()): # 6915 lines in total
        s += line.decode('utf-8')
        if (counter+1)%chunk_size == 0:
            if df_cal.empty:
                df_cal = pd.read_csv(StringIO(s),sep=";",header=None)
                s = ""
            else:
                df_cal = df_cal.append(pd.read_csv(StringIO(s),sep=";",header=None))
                s = ""
                
    df_cal = df_cal.append(pd.read_csv(StringIO(s),sep=";",header=None))
    
    s = ""
    for counter,line in enumerate(zipfile_2019.open("YCAL.CSV").readlines()):
        s += line.decode('utf-8')
        
    df_y_cal = pd.read_csv(StringIO(s),sep=";",header=None)
    df_y_cal.columns = ["Soy oil","Lucerne","Barley"]
    
    
    df_test = pd.DataFrame()
    s = ""
    for counter,line in enumerate(zipfile_2019.open("XVAL.CSV").readlines()): # 600 lines in total
        s += line.decode('utf-8')
        if (counter+1)%chunk_size == 0:
            if df_test.empty:
                df_test = pd.read_csv(StringIO(s),sep=";",header=None)
                s = ""
            else:
                df_test = df_test.append(pd.read_csv(StringIO(s),sep=";",header=None))
                s = ""
    
    X = df_cal.values
    Y = df_y_cal.values
    XTest = df_test.values
    
    return X,Y,XTest
#%% Read IDRC 2002 data set
def get_IDRC_2002_data(instrument=1):
    """
        Helper function to collect the IDRC 2002 shootout data set
        The data set consist of 
            155 Training samples
            40 Validation samples
            460 Test samples
        
        Each sample is measured on two instruments (1 and 2). Only the samples from one instrument will be returned.
        The function collect the training and validation samples into one pool.
        
        The response, y, has three columns:
            1: Weight (g)
            2: Hardness
            3: Assay (g)
        
        The target is to predict the assay column. The data is therefore converted into assay w/w %.
                
        The function takes the input
            instrument: Instrument number. Valid inputs are 1 (default) or 2
        
        The function returns the tuple X,Y,XTest,YTest
            X:          Training samples (spectra)              (195 x 680)
            Y:          Reference values for training samples   (195 x 1)
            XTest:      Test samples (spectra)                  (460 x 680)
            YTest:      Reference values for the test samples   (460 x 1)

    """
    
    if not str(instrument) in ["1","2"]:
        print("You did not provide a valid instrument.")
        print("The default instrument 1 will be used.")
        instrument = 1
    
    from scipy.io import loadmat
    import numpy as np
    
    url_2002 = "https://eigenvector.com/wp-content/uploads/2019/06/nir_shootout_2002.mat_.zip"
    response_2002 = requests.get(url_2002)
    zipfile_2002 = ZipFile(BytesIO(response_2002.content))
    
    dat_2002 = loadmat(BytesIO(zipfile_2002.read("nir_shootout_2002.mat")),mat_dtype=False)
    
    XCal = dat_2002["calibrate_"+str(instrument)][0,0][5]
    YCal = dat_2002["calibrate_Y"][0,0][5]
    
    YCal = YCal[:,2]/YCal[:,0]*100
    
    XVal = dat_2002["validate_"+str(instrument)][0,0][5]
    YVal = dat_2002["validate_Y"][0,0][5]
    
    YVal = YVal[:,2]/YVal[:,0]*100
    
    XTest = dat_2002["test_"+str(instrument)][0,0][5]
    YTest = dat_2002["test_Y"][0,0][5]
    
    YTest = YTest[:,2]/YTest[:,0]*100
    YTest = YTest.reshape(-1,1)
    
    X = np.vstack((XCal,XVal))
    Y = np.concatenate((YCal,YVal)).reshape(-1,1)

    return X,Y,XTest,YTest

#%% Read SWRIdata set
def get_SWRI_data(idxY=5):
    """
        Helper function to collect the SWRI diesel data set
        The data set consist of 784 samples. There are 7 different properties measured for this data set. However, they are not mearsured for all samples. 
        
        The properties are
            bp50:   Boiling point at 50% recovery, deg C (ASTM D 86)
            CN:     Cetane Number (like Octane number only for diesel, ASTM D 613)
            d4052:  Density, g/mL, @ 15 deg C, (ASTM D 4052)
            flash:  Unknown
            freeze: Freezing temperature of the fuel, deg C
            total:  Total aromatics, mass% (ASTM D 5186)
            visc:   Viscosity, cSt, @ 40 deg C

        The function takes the input
            idxY: property id
        
        The function returns the tuple X,Y,XTest,YTest
            X:          Training samples (spectra)              
            Y:          Reference values for training samples   

    """
    
    if not str(idxY) in [str(i) for i in range(7)]:
        print("You did not provide a valid property id.")
        print("The default property id 5 will be used.")
        idxY = 5
    
    from scipy.io import loadmat
    import numpy as np
    
    url_SWRI = "https://eigenvector.com/wp-content/uploads/2019/06/SWRI_Diesel_NIR.zip"

    response_SWRI = requests.get(url_SWRI)
    zipfile_SWRI = ZipFile(BytesIO(response_SWRI.content))
    
    dat_SWRI = loadmat(BytesIO(zipfile_SWRI.read("SWRI_Diesel_NIR.mat")),mat_dtype=False)
    
    Y_all = dat_SWRI["diesel_prop"][0,0][7].astype(np.float64)
    X_all = dat_SWRI["diesel_spec"][0,0][7].astype(np.float64)
    
    idxSamples = np.where(1-np.isnan(Y_all[:,idxY]))[0]
    
    X = X_all[idxSamples]
    Y = Y_all[idxSamples,idxY].reshape(-1,1)
    
    return X,Y

#%%
def dataaugment(x, y, muX=None, sigmaX=None, do_shuffle=False, nAugment=5, betashift=0.1, slopeshift=0.1, multishift=0.1):
    """
        Wrapper function for scatteraugment
    """
    if len(x.shape) == 1:
        x = x.reshape(1,-1)
    if isinstance(muX,type(None)):
        muX = x.mean()
    if isinstance(sigmaX,type(None)):
        sigmaX = x.std()
    
    x_aug = np.repeat(x,nAugment,axis=0)
    
    x_aug = scatteraugment((x_aug-muX)/(2*sigmaX), betashift=betashift, slopeshift=slopeshift, multishift=multishift)*2*sigmaX + muX
    y_aug = np.repeat(y,nAugment,axis=0)
        
    if do_shuffle:
        x_aug,y_aug, = shuffle(x_aug,y_aug)
    
    return x_aug,y_aug
        
#%%
def scatteraugment(x, betashift=0.1, slopeshift=0.1,multishift=0.1):#,nAugment = 5):
    """
        Fetched at https://github.com/EBjerrum/Deep-Chemometrics/blob/master/ChemUtils.py and renamed
    """
    if len(x.shape) == 1:
        x = x.reshape(1,-1)
    
    #Shift of baseline
    #calculate arrays    
    beta = np.random.random(size=(x.shape[0],1))*2*betashift-betashift
    slope = np.random.random(size=(x.shape[0],1))*2*slopeshift-slopeshift + 1
    #Calculate relative position
    axis = np.array(range(x.shape[1]))/float(x.shape[1])
    #Calculate offset to be added
    offset = slope*(axis) + beta - axis - slope/2. + 0.5
    #Multiplicative
    multi = np.random.random(size=(x.shape[0],1))*2*multishift-multishift + 1

    x = multi*x + offset

    return x