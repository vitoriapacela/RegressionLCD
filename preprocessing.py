'''
preprocessing.py
Contains custom utilities for preprocessing raw data.
Author: Vitoria Barin Pacela
e-mail: vitoria.barimpacela@helsinki.fi
'''

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
    Naive sum of the shower deposits in the HCAL and ECAL.
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

def genSum(generator=gen_from_data, train_dir, batch_size=500, data_keys=[["ECAL", "HCAL"], "target"], prep_func=reshapeData):
    '''
    Returns the generator (ECAL and HCAL raw input) and the naive sum of energies in the ECAL and HCAL.
    :param generator:
    :param train_dir:
    :param batch_size:
    :param data_keys:
    :param prep_func:
    :return: ECAL, HCAL, sum_ecal, sum_hcal
    '''
    s_ecal = 0
    s_hcal = 0

    ecal, hcal = generator(train_dir, batch_size, data_keys, prep_func)

    s_ecal = np.sum(ecal)
    s_hcal = np.sum(hcal)

    return ecal, hcal, s_ecal, s_hcal


def nSamples(directory):
    from CMS_Deep_Learning.io import nb_samples_from_h5

    samples = 0
    for f in os.listdir(directory):
        samples += nb_samples_from_h5(directory+f)

    return samples