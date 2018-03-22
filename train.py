import os
import sys
import glob
import re
import h5py
os.environ['KERAS_BACKEND'] = 'tensorflow'
import setGPU
from keras.callbacks import EarlyStopping, ModelCheckpoint

if __package__ is None:
    sys.path.append(os.path.realpath("/data/shared/Software/CMS_Deep_Learning"))
    
from CMS_Deep_Learning.io import gen_from_data, retrieve_data

if __package__ is None:
    sys.path.append(os.path.realpath("/data/shared/Software/RegressionLCD"))
    
from model import *
from preprocessing import *
from analysis import *

mName = 'fix_chPi_dnn'

model = dnnModel(modName = mName)

# Defining the directories, which contain the split data

train_dir = "/bigdata/shared/LCD2018/ChPiEscan/train/"
valid_dir = "/bigdata/shared/LCD2018/ChPiEscan/val/"
#test_dir = "/bigdata/shared/LCD2018/ChPiEscan/test/"


# generator
# training set:
train = gen_from_data(train_dir, batch_size=400, data_keys=[["ECAL", "HCAL"], "energy"])

# validation set:
val = gen_from_data(valid_dir, batch_size=400, data_keys=[["ECAL", "HCAL"], "energy"])

# testing set:
test = gen_from_data(valid_dir, batch_size=400, data_keys=[["ECAL", "HCAL"], "energy"])

hist = model.fit_generator(train, 
                           samples_per_epoch=tr_samples,
                           nb_epoch=50,
                           validation_data = val, 
                           nb_val_samples=val_samples,
                           verbose=1,
                           callbacks=[EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='min')
                           , ModelCheckpoint(filepath=('/nfshome/vitoriabp/gpu-4-culture-plate-sm/new_ds_notebooks/' + mName + '.h5'), 
                                             monitor='val_loss', 
                                             verbose=0, 
                                             save_best_only=True
                                             , mode='min'
                                            )]
                            )

saveLosses(hist, name="mName")

show_losses([("chPi", hist)])
