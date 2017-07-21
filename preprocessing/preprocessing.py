import numpy as np
import os
import sys
import re
import glob
import h5py
import os
#os.environ['KERAS_BACKEND'] = 'theano'
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Activation, Input, Dense, Dropout, merge, Reshape, Convolution3D, MaxPooling3D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Danny's generator:

if __package__ is None:
    import sys, os

    sys.path.append(os.path.realpath("/data/shared/Software/CMS_Deep_Learning"))

from CMS_Deep_Learning.io import gen_from_data, retrieve_data


def show_losses(histories):
    '''
    Adapted from Jean-Roch Vlimant's Keras tutorial.
    Plots loss history of the trained model.
    :param histories: array containing history of losses.
    '''
    plt.figure(figsize=(5, 5))
    # plt.ylim(bottom=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Error by Epoch')
    colors = []
    do_acc = False

    for label, loss in histories:
        color = tuple(np.random.random(3))
        colors.append(color)
        l = label
        vl = label + " validation"

        if 'acc' in loss.history:
            l += ' (acc %2.4f)' % (loss.history['acc'][-1])
            do_acc = True

        if 'val_acc' in loss.history:
            vl += ' (val acc %2.4f)' % (loss.history['val_acc'][-1])
            do_acc = True
        plt.plot(loss.history['loss'], label=l, color=color)

        if 'val_loss' in loss.history:
            plt.plot(loss.history['val_loss'], lw=2, ls='dashed', label=vl, color=color)

    plt.legend()
    plt.yscale('log')
    plt.show()

    if not do_acc: return

    plt.figure(figsize=(5, 5))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    for i, (label, loss) in enumerate(histories):
        color = colors[i]
        if 'acc' in loss.history:
            plt.plot(loss.history['acc'], lw=2, label=label + " accuracy", color=color)

        if 'val_acc' in loss.history:
            plt.plot(loss.history['val_acc'], lw=2, ls='dashed', label=label + " validation accuracy", color=color)

    plt.legend(loc='lower right')
    plt.show()


def reshapeData(inp):
    '''
    Function to reshape the data, parameter to Danny's generator
    :param inp: array containing ECAL (shape (10000, 25, 25, 25)), HCAL (shape (10000, 5, 5, 60)) and target data (shape (10000, 2)).
    :return: ECAL, HCAL and target arrays in the right shape.
    '''
    (xe, xh), y = inp
    energy = [y[:, 1:]]
    return (xe, xh), energy