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