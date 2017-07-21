'''
plots.py
Contains custom utilities for plotting and displaying tested data.
Author: Vitoria Barin Pacela
e-mail: vitoria.barimpacela@helsinki.fi
'''

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
from scipy.stats import norm


def histEDif(difference, nbins=50, lim=25, lim_l=0, lim_r=550):
    '''
    This function plots histogram of the difference between the predicted energy and the target energy (GeV).

    Parameters:
    difference - energy difference as numpy.ndarray. Suggestion: use dif() function here.
    nbins - number of bins desired for the histogram, as integer.
    lim - integer defining the minimum and maximum values for the x axis.
    lim_l - integer defining the minimum energy value, used in the title of the plot.
    lim_r - integer defining the maximum energy value, used in the title of the plot.
    '''

    # the histogram of the data
    n, bins, patches = plt.hist(difference, nbins, normed=1, facecolor='green', alpha=0.75)

    # best fit of data
    (mu, sigma) = norm.fit(difference)

    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=1)

    plt.xlabel('Difference between the predicted energy and the target energy (GeV)')
    plt.ylabel('Probability')
    plt.title("Energy difference in the regression model, energies between %d and %d GeV" % (lim_l, lim_r))

    plt.xlim(-lim, lim)

    plt.show()
    #plt.savefig("histDif_%d_%d.jpg" % (lim_l, lim_r))


def histRelDif(difference, nbins=50, lim=0.2, lim_l=0, lim_r=550):
    '''Plots histogram of the normalized energy difference (%).

    Parameters:
    difference - energy difference as numpy.ndarray. Suggestion: use dif() function here.
    nbins - number of bins desired for the histogram, as integer.
    lim - integer defining the minimum and maximum values for the x axis.
    lim_l - integer defining the minimum energy value, used in the title of the plot.
    lim_r - integer defining the maximum energy value, used in the title of the plot.
    '''

    # the histogram of the data
    n, bins, patches = plt.hist(difference, nbins, normed=1, facecolor='green', alpha=0.75)

    mean = np.mean(difference)
    print mean

    std = np.std(difference)  # standard deviation
    print std

    # best fit of data
    (mu, sigma) = norm.fit(difference
                           # , range = (-std,std)
                           )
    print mu, sigma

    # add a "best fit" line
    y = mlab.normpdf(bins, mu, sigma)  # normal probability density function
    l = plt.plot(bins, y, 'r--', linewidth=1)

    plt.xlim(-lim, lim)
    plt.xlabel('Relative difference between the predicted energy and the target energy (%)')
    plt.ylabel('Probability')
    plt.title("Relative energy difference in the regression model, energies between %d and %d GeV" % (lim_l, lim_r))

    plt.show()


def plotRelXTarget(target, relative, lim_l=0, lim_r=500):
    '''
    Plots the relative energy difference against the target energy.

    Parameters:
    target - energy target (label) array (numpy.ndarray)
    relative - relative energy difference as numpy.ndarray.
    lim_l - (int, double, float) defines the minimum value in the x axis.
    lim_r - (int, double, float) defines the maximum value in the x axis.
    '''

    plt.figure(figsize=(5, 5))
    plt.xlabel("Target energy (GeV)")
    plt.ylabel("Relative energy difference (%)")
    plt.title("Testing set: Comparing the relative energy difference with the target energy of the particle")

    plt.scatter(target, relative, color='g', alpha=0.5)

    plt.xlim(lim_l, lim_r)
    plt.legend()
    plt.show()


def plotMeanXEnergy(target, predicted, lim_y=0.14, lim_l=0, lim_r=520, limit=False):
    '''
    Plots the prediction error.
    (Mean of the distribution of the difference between the target and the predicted energy, divided by the prediction energy, against the target energy).

    Parameters:
    target - energy target (label) array (numpy.ndarray)
    predicted - array (numpy.ndarray) of predictions from testing
    lim_y - integer defining the maximum value for the y axis.
    lim_l - integer defining the minimum energy value, x axis.
    lim_r - integer defining the maximum energy value, x axis.
    limit - boolean that defines whether or not to limit the axes.
    '''
    plt.figure(figsize=(5, 5))
    plt.xlabel("True energy (GeV)")
    plt.ylabel("Energy difference mean / Predicted energy")
    plt.title("Mean, energies between %d and %d GeV" % (lim_l, lim_r))

    plt.scatter(target, np.mean(dif(target, predicted)) / predicted, color='g', alpha=0.5)

    if (limit):
        plt.xlim(lim_l, lim_r)
        plt.ylim(0, lim_y)

    plt.legend()
    plt.show()


def plotStdXEnergy(target, predicted, lim_y=2.7, lim_l=0, lim_r=520, limit=False):
    '''
    Plots the prediction deviation depending on the true energy.
    (Standard deviation of the distribution of the difference between the target and the predicted energy, divided by the predicted energy, against the target energy).

    Parameters:
    target - energy target (label) array (numpy.ndarray)
    predicted - array (numpy.ndarray) of predictions from testing
    lim_y - integer defining the maximum value for the y axis.
    lim_l - integer defining the minimum energy value, x axis.
    lim_r - integer defining the maximum energy value, x axis.
    limit - boolean that defines whether or not to limit the axes.
    '''
    plt.figure(figsize=(5, 5))
    plt.xlabel("True energy (GeV)")
    plt.ylabel("Standard deviation of the energy difference / Predicted energy")
    plt.title("Standard deviation, energies between %d and %d GeV" % (lim_l, lim_r))

    plt.scatter(target, np.std(dif(target, predicted)) / predicted, color='g', alpha=0.5)

    if limit:
        plt.xlim(lim_l, lim_r)
        plt.ylim(0, lim_y)

    plt.legend()
    plt.show()


def plotMeans():
    mean1 = np.mean(dif(x[0], y[0]))
    mean2 = np.mean(dif(x[1], y[1]))
    mean3 = np.mean(dif(x[2], y[2]))
    mean4 = np.mean(dif(x[3], y[3]))
    mean5 = np.mean(dif(x[4], y[4]))

    plt.figure(figsize=(5, 5))
    plt.xlabel("Energy bin")
    plt.ylabel("Energy difference mean (GeV)")
    plt.title("Mean")

    plt.scatter(1, mean1, color='green', alpha=0.5, label='0-100 GeV')
    plt.scatter(2, mean2, color='red', alpha=0.5, label='100-200 GeV')
    plt.scatter(3, mean3, color='purple', alpha=0.5, label='200-300 GeV')
    plt.scatter(4, mean4, color='blue', alpha=0.5, label='300-400 GeV')
    plt.scatter(5, mean5, color='orange', alpha=0.5, label='400-500 GeV')

    plt.legend(loc=2)
    plt.show()
    # plt.savefig("mean_dif.jpg")


def plotStds():
    std1 = np.std(dif(x[0], y[0]))
    std2 = np.std(dif(x[1], y[1]))
    std3 = np.mean(dif(x[2], y[2]))
    std4 = np.std(dif(x[3], y[3]))
    std5 = np.std(dif(x[4], y[4]))

    plt.figure(figsize=(5, 5))
    plt.xlabel("Energy bin")
    plt.ylabel("Energy difference standard deviation (GeV)")
    plt.title("Standard deviation")

    plt.scatter(1, std1, color='green', alpha=0.5, label='0-100 GeV')
    plt.scatter(2, std2, color='red', alpha=0.5, label='100-200 GeV')
    plt.scatter(3, std3, color='purple', alpha=0.5, label='200-300 GeV')
    plt.scatter(4, std4, color='blue', alpha=0.5, label='300-400 GeV')
    plt.scatter(5, std5, color='orange', alpha=0.5, label='400-500 GeV')

    plt.ylim(0, 8.5)

    plt.legend(loc=2)
    plt.show()
    # plt.savefig("std_dif.jpg")


def plotRMeans():
    mean1 = np.mean(rDif(x[0], y[0]))
    mean2 = np.mean(rDif(x[1], y[1]))
    mean3 = np.mean(rDif(x[2], y[2]))
    mean4 = np.mean(rDif(x[3], y[3]))
    mean5 = np.mean(rDif(x[4], y[4]))

    plt.figure(figsize=(5, 5))
    plt.xlabel("Energy bin")
    plt.ylabel("Relative energy difference mean (%)")
    plt.title("Mean")

    plt.scatter(1, mean1, color='green', alpha=0.5, label='0-100 GeV')
    plt.scatter(2, mean2, color='red', alpha=0.5, label='100-200 GeV')
    plt.scatter(3, mean3, color='purple', alpha=0.5, label='200-300 GeV')
    plt.scatter(4, mean4, color='blue', alpha=0.5, label='300-400 GeV')
    plt.scatter(5, mean5, color='orange', alpha=0.5, label='400-500 GeV')

    plt.ylim(-0.003, 0.003)

    plt.legend(loc=2)
    plt.show()
    # plt.savefig("mean_rDif.jpg")


def plotRStds():
    std1 = np.std(rDif(x[0], y[0]))
    std2 = np.std(rDif(x[1], y[1]))
    std3 = np.mean(rDif(x[2], y[2]))
    std4 = np.std(rDif(x[3], y[3]))
    std5 = np.std(rDif(x[4], y[4]))

    plt.figure(figsize=(5, 5))
    plt.xlabel("Energy bin")
    plt.ylabel("Relative energy difference standard deviation (%)")
    plt.title("Standard deviation")

    plt.scatter(1, std1, color='green', alpha=0.5, label='0-100 GeV')
    plt.scatter(2, std2, color='red', alpha=0.5, label='100-200 GeV')
    plt.scatter(3, std3, color='purple', alpha=0.5, label='200-300 GeV')
    plt.scatter(4, std4, color='blue', alpha=0.5, label='300-400 GeV')
    plt.scatter(5, std5, color='orange', alpha=0.5, label='400-500 GeV')

    plt.ylim(0, 0.1)

    plt.legend(loc='best')
    plt.show()
    # plt.savefig("std_rDif.jpg")