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
    #plt.savefig("loss.jpg")


def dif(target, predicted):
    '''
    Returns the difference between the target energy and the predicted energy.
    :param target: energy target (true value label) as numpy.ndarray.
    :param predicted: numpy.ndarray of predictions from testing.
    :return: energy difference as numpy.ndarray.
    '''

    dif = target - predicted
    dif = dif.reshape((dif.shape[0],))
    return dif


def rDif(target, predicted):
    '''
    Returns the relative difference to the target energy of the particle.
    :param target: energy target (true value label) as numpy.ndarray.
    :param predicted: numpy.ndarray of predictions from testing.
    :return: relative energy difference as numpy.ndarray.
    '''

    dif = target - predicted
    rDif = dif / target
    rDif = rDif.reshape((rDif.shape[0],))
    return rDif


def plotPredictedXTarget(target, predicted, lim_l=0, lim_r=550):
    '''
    Plots the predicted energy against the target energy.
    :param target: energy target (true value label) as numpy.ndarray.
    :param predicted: numpy.ndarray of predictions from testing.
    :param lim_l: (int, float) defines the minimum value for both the x and y axes.
    :param lim_r: (int, float) defines the maximum value for both the x and y axes.
    '''

    plt.figure(figsize=(5, 5))
    plt.xlabel("Target energy (GeV)")
    plt.ylabel("Predicted energy (GeV)")
    plt.title("Predicted energy against the true energy of the particle, energies between %d and %d GeV" % (lim_l, lim_r))

    plt.scatter(target, predicted, color='g', alpha=0.5)

    plt.xlim(lim_l, lim_r)
    plt.ylim(lim_l, lim_r)

    plt.legend()
    plt.show()

def histEDif(difference, nbins=50, lim=25, lim_l=0, lim_r=550):
    '''
    This function plots histogram of the difference between the predicted energy and the target energy (GeV).
    :param difference: energy difference as numpy.ndarray. Suggestion: use dif() function here.
    :param nbins: number of bins desired for the histogram, as integer.
    :param lim: (int, float) defines the minimum and maximum values for the x axis.
    :param lim_l: (int, float) defines the minimum energy value, used in the title of the plot.
    :param lim_r: (int, float) defines the maximum energy value, used in the title of the plot.
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
    '''
    Plots histogram of the normalized energy difference (%).
    :param difference: energy difference as numpy.ndarray. Suggestion: use dif() function here.
    :param nbins: number of bins desired for the histogram, as integer.
    :param lim: (int, float) defines the minimum and maximum values for the x axis.
    :param lim_l: (int, float) defines the minimum energy value, used in the title of the plot.
    :param lim_r: (int, float) defines the maximum energy value, used in the title of the plot.
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
    # plt.savefig("histRDif_%d_%d.jpg" % (lim_l, lim_r))


def plotRelXTarget(target, relative, lim_l=0, lim_r=500):
    '''
    Plots the relative energy difference against the target energy.
    :param target: energy target (true value label) as numpy.ndarray.
    :param relative: relative energy difference as numpy.ndarray.
    :param lim_l: (int, float) defines the minimum value in the x axis.
    :param lim_r: (int, float) defines the maximum value in the x axis.
    '''

    plt.figure(figsize=(5, 5))
    plt.xlabel("Target energy (GeV)")
    plt.ylabel("Relative energy difference (%)")
    plt.title("Testing set: Comparing the relative energy difference with the target energy of the particle")

    plt.scatter(target, relative, color='g', alpha=0.5)

    plt.xlim(lim_l, lim_r)
    plt.legend()
    plt.show()
    # plt.savefig("relXtarget_%d_%d.jpg" % (lim_l, lim_r))


def plotMeanXEnergy(target, predicted, lim_y=0.14, lim_l=0, lim_r=520, limit=False):
    '''
    Plots the prediction error.
    (Mean of the distribution of the difference between the target and the predicted energy,
     divided by the prediction energy, against the target energy).
    :param target: energy target (true value label) as numpy.ndarray.
    :param predicted: numpy.ndarray of predictions from testing.
    :param lim_y: (int, float) defines the maximum value for the y axis.
    :param lim_l: (int, float) defines the minimum energy value, x axis.
    :param lim_r: (int, float) defines the maximum energy value, x axis.
    :param limit: boolean that defines whether or not to limit the axes.
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
    # plt.savefig("meanXenergy_%d_%d.jpg" % (lim_l, lim_r))


def plotStdXEnergy(target, predicted, lim_y=2.7, lim_l=0, lim_r=520, limit=False):
    '''
    Plots the prediction deviation depending on the true energy.
    (Standard deviation of the distribution of the difference between the target and the predicted energy,
    divided by the predicted energy, against the target energy).
    :param target: energy target (true value label) as numpy.ndarray.
    :param predicted: numpy.ndarray of predictions from testing.
    :param lim_y: (int, float) defines the maximum value for the y axis.
    :param lim_l: (int, float) defines the minimum energy value, x axis.
    :param lim_r: (int, float) defines the maximum energy value, x axis
    :param limit: boolean that defines whether or not to limit the axes.
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
    # plt.savefig("stdXenergy_%d_%d.jpg" % (lim_l, lim_r))


def plotMeans():
    '''
    Plots the mean of 5 energy bins in GeV.
    '''
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
    # plt.savefig("means.jpg")


def plotStds():
    '''
    Plots the standard deviation of 5 energy bins in GeV.
    '''
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
    # plt.savefig("stds.jpg")


def plotRMeans():
    '''
    Plots the relative mean of 5 energy bins in %.
    :return:
    '''
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
    # plt.savefig("means_rDif.jpg")


def plotRStds():
    '''
    Plots the relative standard deviation of 5 energy bins in %.
    '''
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