import os
import sys
import glob
import re
import h5py
os.environ['KERAS_BACKEND'] = 'tensorflow'
#import setGPU
#from keras.callbacks import EarlyStopping, ModelCheckpoint

if __package__ is None:
    sys.path.append(os.path.realpath("/data/shared/Software/CMS_Deep_Learning"))
    
#from CMS_Deep_Learning.io import gen_from_data, retrieve_data
#from CMS_Deep_Learning.postprocessing.metrics import distribute_to_bins
from CMS_Deep_Learning.io import simple_grab

if __package__ is None:
    sys.path.append(os.path.realpath("/data/shared/Software/RegressionLCD"))
    
from model import *
from preprocessing import *
from analysis import *

mName = 'fix_chPi_dnn'

test_dir = "/bigdata/shared/LCD2018/ChPiEscan/test/"

os.environ['CUDA_VISIBLE_DEVICES'] = ''

mod = ('/nfshome/vitoriabp/gpu-4-culture-plate-sm/new_ds_notebooks/' + mName + '.json') # model file
w = ('/nfshome/vitoriabp/gpu-4-culture-plate-sm/new_ds_notebooks/' + mName + '.h5') # weights file

# grab y and predictions together
all_y, all_pred, = simple_grab(['Y', 'predictions'], data = test_dir, label_keys='energy',
                               input_keys=['ECAL', 'HCAL'],
                               model = mod,
                               weights = w)



all_pred = np.reshape(all_pred, (all_pred[0],))
                      
#print(all_y.shape)
#print(all_pred.shape)

saveTruePred(name=mName, true=all_y, pred=all_pred)

histEDif(target=all_y, pred=all_pred)

histRelDif(target=all_y, pred=all_pred)

x, y, means, rMeans, stds, rStds, sizes, res = binning(10, all_y.ravel(), all_pred.ravel())

plotN(means, stds, sizes, "means")

plotN(rMeans, rStds, sizes, "rMeans")

plotN(rStds, rStds, sizes, "rStds")
