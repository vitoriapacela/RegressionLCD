'''
plots.py
Contains custom utilities for creating DNN models.
Author: Vitoria Barin Pacela
e-mail: vitoria.barimpacela@helsinki.fi
'''

import numpy as np
import root_numpy as rnp
import os
import sys
import re
import glob
import h5py
import numpy as np
import os
#os.environ['KERAS_BACKEND'] = 'theano'
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Activation, Input, Dense, Dropout, merge, Reshape, Convolution3D, MaxPooling3D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint


def loadModel(name, weights=False):
    '''
    Adapted from Kaustuv Datta and Jayesh Mahapatra's CaloImageMacros.
    Loads models from json file.
    :param name: (String) name of the json file.
    :param weights: (boolean) whether or not to load the weights.
    :return: loaded model.
    '''

    json_file = open('%s.json' % name, 'r')
    loadedmodeljson = json_file.read()
    json_file.close()

    model = model_from_json(loadedmodeljson)

    # load weights into new model
    if weights == True:
        model.load_weights('%s.h5' % name)
    # print model.summary()

    print("Loaded model from disk")
    return model


def saveModel(model, name="regression"):
    '''
    Saves model as json file.
    Adapted from Kaustuv Datta and Jayesh Mahapatra's CaloImageMacros.
    :param model: model to be saved.
    :param name: (String) name of the model to be saved.
    :return: saved model.
    '''
    model_name = name
    model.summary()
    model.save_weights('%s.h5' % model_name, overwrite=True)
    model_json = model.to_json()
    with open("%s.json" % model_name, "w") as json_file:
        json_file.write(model_json)


def saveLosses(hist, name="regression"):
    '''
    Adapted from Kaustuv Datta and Jayesh Mahapatra's CaloImageMacros.
    Saves model losses into an HDF5 file.
    :param hist: array of losses in the trained model.
    :param name: (String) name of the file to be saved
    '''
    loss = np.array(hist.history['loss'])
    valoss = np.array(hist.history['val_loss'])

    f = h5py.File("%s_losses.h5" % name, "w")
    f.create_dataset('loss', data=loss)
    f.create_dataset('val_loss', data=valoss)
    f.close()