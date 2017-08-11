'''
preprocessing.py
Contains custom utilities for preprocessing raw data.
Author: Vitoria Barin Pacela
e-mail: vitoria.barimpacela@helsinki.fi
'''
import os, sys, glob, h5py

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


def nSum(directory):
    '''
    Naive sum of the shower deposits in the ECAL and HCAL.
    :param directory: path to the directory with HDF5 files.
    :return: sum of the energies in the ECAL and HCAL, respectively.
    '''


    s_ecal = 0
    s_hcal = 0

    if (os.path.exists(os.path.abspath(directory)) and os.path.isdir(directory)):
        directory = glob.glob(os.path.abspath(directory) + "/*.h5")

    for fileName in directory:
        inp = h5py.File(fileName, "r")
        ecal = np.array(inp["ECAL"], dtype="float32")
        hcal = np.array(inp["HCAL"], dtype="float32")

        s_ecal += np.sum(ecal)
        s_hcal += np.sum(hcal)

    return s_ecal, s_hcal


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
    :return: ECAL, HCAL, HCALsum
    :rtype: numpy array with shape (n, 25, 25, 25), array with shape (n, 5, 5, 60), array with shape (n, 1); n is the batch size.
    '''
    while True:
        (ecal, hcal), true = next(generator)
        s_hcal = np.sum(np.sum(np.sum(hcal, axis=-1), axis=-1), axis=-1, keepdims=True)
        yield [ecal, hcal, s_hcal], true


def _genSum(generator):
    '''
    -- This function might be buggy, haven't tested yet. --
    Generator that receives a generator (Danny's) and outputs ECAL, HCAL and an array containing the sum over the ECAL and HCAL cells.
    :param generator: gen_from_data(train_dir, batch_size=500, data_keys=[["ECAL", "HCAL"], "target"], prep_func=reshapeData)
    :type generator: generator
    :return: ECAL, HCAL, sum[ECAL, HCAL]
    :rtype: numpy array with shape (n, 25, 25, 25), array with shape (n, 5, 5, 60), array with shape (n, 2); n is the batch size.
    '''
    # s = lambda x: np.sum(np.sum(np.sum(x,axis=-1),axis=-1),axis=-1)
    s = lambda x: np.sum(np.sum(np.sum(x, axis=-1), axis=-1), axis=-1, keepdims=True)

    while True:
        (ecal, hcal), true = next(generator)
        s_ecal = s(ecal)
        s_hcal = s(hcal)
        sums = np.array([s_ecal, s_hcal])
        reshaped = sums.reshape(500,2)
        # print reshaped.shape
        # print reshaped
        yield [ecal, hcal, sums], true