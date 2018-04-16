'''
preprocessing.py
Contains custom utilities for preprocessing raw data.
Author: Vitoria Barin Pacela
e-mail: vitoria.barimpacela@helsinki.fi
'''
import os, sys, glob, h5py
import numpy as np

if __package__ is None:
    sys.path.append(os.path.realpath("/data/shared/Software/CMS_Deep_Learning"))
from CMS_Deep_Learning.io import simple_grab, nb_samples_from_h5

def reshapeData(inp):
    '''
    Function to reshape the data, parameter to Danny's generator
    :param inp: array containing ECAL (shape (10000, 25, 25, 25)), HCAL (shape (10000, 5, 5, 60)) and target data (shape (10000, 2)).
    :type inp: array
    :return: ECAL, HCAL and target arrays in the right shape.
    '''
    (xe, xh), y = inp
    energy = [y[:, 1:]]
    return (xe, xh), energy


# def nSum(directory):
#     '''
#     Does not work. Check out new function.
#     Naive sum of the shower deposits in the ECAL and HCAL.
#     :param directory: path to the directory with HDF5 files.
#     :return: sum of the energies in the ECAL and HCAL, respectively.
#     '''
#
#
#     s_ecal = 0
#     s_hcal = 0
#
#     if (os.path.exists(os.path.abspath(directory)) and os.path.isdir(directory)):
#         directory = glob.glob(os.path.abspath(directory) + "/*.h5")
#
#     for fileName in directory:
#         inp = h5py.File(fileName, "r")
#         ecal = np.array(inp["ECAL"], dtype="float32")
#         hcal = np.array(inp["HCAL"], dtype="float32")
#
#         s_ecal += np.sum(np.sum(np.sum(ecal, axis=-1), axis=-1), axis=-1, keepdims=True)
#         s_hcal += np.sum(np.sum(np.sum(hcal, axis=-1), axis=-1), axis=-1, keepdims=True)
#
#     return s_ecal, s_hcal


def nSamples(directory):
    '''
    Return number of samples in the directory.
    :param directory: path to directory that contains the HDF5 files
    :type directory: str
    :return: number of samples
    :rtype: int
    '''
    import sys
    import os
    if __package__ is None:
        sys.path.append(os.path.realpath("/data/shared/Software/CMS_Deep_Learning"))
    from CMS_Deep_Learning.io import nb_samples_from_h5

    samples = 0
    for f in os.listdir(directory):
        samples += nb_samples_from_h5(directory+f)

    return samples


def genHsum(generator):
    '''
    Generator that receives a generator (Danny's) and outputs ECAL, HCAL and the sum over the HCAL cells.
    :param generator: gen_from_data(train_dir, batch_size=500, data_keys=[["ECAL", "HCAL"], "target"], prep_func=reshapeData)
    :type generator: generator
    :return: ECAL, HCAL, sum
    :rtype: numpy array with shape (n, 25, 25, 25), array with shape (n, 5, 5, 60), array with shape (n, 1); n is the batch size.
    '''
    while True:
        (ecal, hcal), true = next(generator)
        s_hcal = np.sum(np.sum(np.sum(hcal, axis=-1), axis=-1), axis=-1, keepdims=True)
        yield [ecal, hcal, s_hcal], true


# def _genSum(generator):
#     '''
#     -- This function might be buggy, haven't tested yet. --
#     Generator that receives a generator (Danny's) and outputs ECAL, HCAL and an array containing the sum over the ECAL and HCAL cells.
#     :param generator: gen_from_data(train_dir, batch_size=500, data_keys=[["ECAL", "HCAL"], "target"], prep_func=reshapeData)
#     :type generator: generator
#     :return: ECAL, HCAL, sum[ECAL, HCAL]
#     :rtype: numpy array with shape (n, 25, 25, 25), array with shape (n, 5, 5, 60), array with shape (n, 2); n is the batch size.
#     '''
#     # s = lambda x: np.sum(np.sum(np.sum(x,axis=-1),axis=-1),axis=-1)
#     s = lambda x: np.sum(np.sum(np.sum(x, axis=-1), axis=-1), axis=-1, keepdims=True)
#
#     while True:
#         (ecal, hcal), true = next(generator)
#         s_ecal = s(ecal)
#         s_hcal = s(hcal)
#         sums = np.array([s_ecal, s_hcal])
#         reshaped = sums.reshape(500,2)
#         # print(reshaped.shape)
#         # print(reshaped)
#         yield [ecal, hcal, sums], true


def sumCal(cal):
    '''
    Sum of the energy deposits over the calorimeter.
    :type cal: numpy.ndarray, 4D.
    :param cal: ECAL or HCAL input.
    :return: sum of the energy values
    :rtype: numpy.ndarray, 2D.
    '''
    s_cal = np.sum(np.sum(np.sum(cal, axis=-1), axis=-1), axis=-1, keepdims=True)
    return s_cal


def inpSum(dir):
    '''
    Naive sum of the shower deposits in the ECAL and HCAL.
    :type dir: str.
    :param dir: path to the directory with HDF5 files.
    :return: sum of the ECAL and HCAL sums.
    :rtype: numpy.ndarray, shape: (n,)
    '''
    # grab ECAL and HCAL inputs
    ecal, hcal = simple_grab('X', data=dir, label_keys=['ECAL', 'HCAL'], input_keys=['ECAL', 'HCAL'])

    # sums
    s_ecal = sumCal(ecal)
    s_hcal = sumCal(hcal)

    # reshape sum output
    s_ecal = s_ecal.ravel()
    s_hcal = s_hcal.ravel()

    # total sum
    inSum = s_ecal + s_hcal
    return inSum


def preSum(train_dir, particle="", reshape=False, label='energy'):
    '''
    To be used before training for data visualization.
    Naive sum of the shower deposits in the ECAL and HCAL.
    Use label='target' in the old dataset.
    :type train_dir: str.
    :parameter train_dir: path to the training directory with HDF5 files.
    :type particle: str.
    :parameter particle: name of the particle.
    :return: energy targets and energy sum arrays.
    :rtype: numpy.ndarray, numpy.ndarray; shape: (n,) shape: (n,)
    '''
    # grab targets (y)
    all_y = simple_grab('Y', data=train_dir, label_keys=label,
                        input_keys=['ECAL', 'HCAL'])
    if(reshape == True):
        all_y = all_y[:, 1:]
        all_y = all_y.ravel()
        #print(all_y.shape)

    # sum of ECAL and HCAL
    inSum = inpSum(train_dir)

    # save arrays to HDF5
    saveSum_toHDF5(particle, all_y, inSum)


    return all_y, inSum


def saveSum_toHDF5(name, true, inSum):
    '''
    Saves true energy and prediction energy arrays into an HDF5 file.
    :parameter name: name of the file to be saved.
    :type name: str.
    :parameter true: array of energy targets (true value label).
    :type true: numpy.ndarray
    :parameter inSum: array of predictions from testing.
    :type inSum: numpy.ndarray
    '''
    true_sum = np.array([true, inSum])
    # best implementation would be to check if the file exists. if not, create it. being lazy for now.
    change_file = h5py.File(name + "TruePred.h5", 'a')
    ds = change_file.create_dataset("true_sum", data=true_sum)
    change_file.close()
