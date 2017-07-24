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