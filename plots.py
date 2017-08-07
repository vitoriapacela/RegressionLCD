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
import os, sys

def show_losses(histories):
    '''
    Adapted from Jean-Roch Vlimant's Keras tutorial.
    Plots loss history of the trained model.
    :parameter histories: array containing history of losses.
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
    :parameter target: energy target (true value label) array.
    :type target: numpy.ndarray
    :parameter predicted: array of predictions from testing.
    :type predicted: numpy.ndarray
    :return: energy difference as numpy.ndarray.
    '''

    dif = target - predicted
    dif = dif.reshape((dif.shape[0],))
    return dif


def rDif(target, predicted):
    '''
    Returns the relative difference to the target energy of the particle, in %.
    :parameter target: energy target (true value label) array.
    :type target: numpy.ndarray
    :parameter predicted: array of predictions from testing.
    :type predicted: numpy.ndarray
    :return: relative energy difference as numpy.ndarray.
    '''

    dif = target - predicted
    rDif = (dif / target)*100
    rDif = rDif.reshape((rDif.shape[0],))

    return rDif


def plotPredictedXTarget(target, predicted, lim_l=0, lim_r=550):
    '''
    Plots the predicted energy against the target energy.
    :parameter target: array of energy targets (true value label).
    :type target: numpy.ndarray
    :parameter predicted: array of predictions from testing.
    :type predicted: numpy.ndarray
    :parameter lim_l: minimum value for both the x and y axes.
    :type lim_l: float
    :parameter lim_r: maximum value for both the x and y axes.
    :type lim_r: float
    '''

    plt.figure(figsize=(5, 5))
    plt.xlabel("Target energy (GeV)")
    plt.ylabel("Predicted energy (GeV)")
    plt.title("Predicted X true energy \n Energies between %d and %d GeV" % (lim_l, lim_r))
    #plt.title("Predicted X true energy")

    plt.scatter(target, predicted, color='g', alpha=0.5)

    plt.xlim(lim_l, lim_r)
    plt.ylim(lim_l, lim_r)

    plt.legend()
    plt.show()

    
def histEDif(target, pred, nbins=50, lim=25, lim_l=0, lim_r=550):
    '''
    This function plots histogram of the difference between the target energy and the predicted energy (GeV).
    :parameter target: array of energy targets (true value label).
    :type target: numpy.ndarray
    :parameter pred: array of predictions from testing.
    :type pred: numpy.ndarray
    :parameter nbins: number of bins desired for the histogram.
    :type nbins: int
    :parameter lim: minimum and maximum values for the x axis.
    :type lim: float
    :parameter lim_l: minimum energy value, used in the title of the plot.
    :type lim_l: float
    :parameter lim_r: maximum energy value, used in the title of the plot.
    :type lim_r
    '''

    difference = dif(target, pred)
    
    # the histogram of the data
    
    mean = np.mean(difference)
    print mean

    std = np.std(difference)  # standard deviation
    print std
    
    error=std/np.sqrt(len(target))
    #labels=['Mean: %.2f' % mean, 'Standard deviation: %.2f' % std]
    
    n, bins, patches = plt.hist(difference, nbins, normed=1, facecolor='green', alpha=0.75)
    
    plt.text(3, 0.12, 'Mean: %.2f $\pm$ %.2f \nStd. dev.: %.2f' % (mean, error, std))

    #plt.xlabel('Difference between true and predicted energy (GeV)')
    plt.xlabel(r'$E_{true} - E_{pred}$ (GeV)', size=16)
    plt.ylabel('Probability', size=16)
    plt.title("Energy difference \n Energies between %d and %d GeV" % (lim_l, lim_r), size=16)
    #plt.title("Energy difference")

    plt.xlim(-lim, lim)

    plt.legend(loc='best')
    
    plt.show()
    #plt.savefig("histDif_%d_%d.jpg" % (lim_l, lim_r))
    

def histRelDif(target, pred, nbins=50, lim=0.2, lim_l=0, lim_r=550):
    '''
    Plots histogram of the normalized energy difference (%).
    :parameter target: array of energy targets (true value label).
    :type target numpy.ndarray
    :parameter pred: array of predictions from testing.
    :type pred: numpy.ndarray
    :parameter nbins: number of bins desired for the histogram.
    :type nbins: int
    :parameter lim: minimum and maximum values for the x axis.
    :type lim: float
    :parameter lim_l: minimum energy value, used in the title of the plot.
    :type lim_l: float
    :parameter lim_r: maximum energy value, used in the title of the plot.
    :type lim_r: float
    '''

    difference = rDif(target, pred)
    
    # the histogram of the data
    
    mean = np.mean(difference)
    print mean

    std = np.std(difference)  # standard deviation
    print std
    
    #labels=['Mean: %.2f' % mean, 'Standard deviation: %.2f' % std]
    error=std/np.sqrt(len(target))
    print error
    
    n, bins, patches = plt.hist(difference, nbins, normed=1, facecolor='green', alpha=0.75)

    plt.text(0.015, 20, 'Mean: %.5f $\pm$ %.5f \nStd. dev.: %.2f' % (mean, error, std))
    #plt.text(3, 0.12, 'Mean: %.2f $\pm$ %.2f \nStd. dev.: %.2f' % (mean, error, std))


    plt.xlim(-lim, lim)
    #plt.xlabel('Relative difference between true and predicted energy (%)')
    plt.xlabel(r'$\frac{(E_{true} - E_{pred})}{E_{true}}$ (GeV)', size=18)
    plt.ylabel('Probability', size=16)
    plt.title("Relative energy difference \n Energies between %d and %d GeV" % (lim_l, lim_r), size=16)
    #plt.title("Relative energy difference")
    
    plt.legend(loc='best')
    plt.show()
    # plt.savefig("histRDif_%d_%d.jpg" % (lim_l, lim_r))


def plotRelXTarget(target, relative, lim_l=0, lim_r=500):
    '''
    Plots the relative energy difference against the target energy.
    :parameter target: array of energy targets (true value label).
    :type target: numpy.ndarray
    :parameter relative: array of the relative energy differences.
    :type relative: numpy.ndarray.
    :parameter lim_l: minimum value in the x axis.
    :type lim_l: float
    :parameter lim_r: defines the maximum value in the x axis.
    :type lim_r: float
    '''

    plt.figure(figsize=(5, 5))
    plt.xlabel("Target energy (GeV)")
    plt.ylabel("Relative energy difference (%)")
    #plt.title("Relative energy difference X Target energy")
    plt.title("Relative energy difference X Target energy \n Energies between %d and %d GeV" % (lim_l, lim_r))

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
    :parameter target: array of energy targets (true value label).
    :type target: numpy.ndarray
    :parameter predicted: array of predictions from testing.
    :type predicted: numpy.ndarray
    :parameter lim_y: maximum value for the y axis.
    :type lim_y: float
    :parameter lim_l: minimum energy value, x axis.
    :type lim_l: float
    :parameter lim_r: maximum energy value, x axis.
    :type lim_r: float
    :parameter limit: whether to limit the axes.
    :type limit: bool
    '''

    plt.figure(figsize=(5, 5))
    plt.xlabel("True energy (GeV)")
    plt.ylabel("Energy difference mean / Predicted energy")
    plt.title("Mean. \n Energies between %d and %d GeV." % (lim_l, lim_r))
    #plt.title("Mean")

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

    :parameter target: array of energy targets (true value label).
    :type target: numpy.ndarray
    :parameter predicted: array of predictions from testing.
    :type predicted: numpy.ndarray
    :parameter lim_y: maximum value for the y axis.
    :type lim_y: float
    :parameter lim_l: minimum energy value, x axis.
    :type lim_l: float
    :parameter lim_r: maximum energy value, x axis
    :type lim_r: float
    :parameter limit: whether or not to limit the axes.
    :type limit: bool
    '''

    plt.figure(figsize=(5, 5))
    plt.xlabel("True energy (GeV)")
    plt.ylabel("Standard deviation of the energy difference / Predicted energy")
    plt.title("Standard deviation \n Energies between %d and %d GeV" % (lim_l, lim_r))
    #plt.title("Standard deviation")

    plt.scatter(target, np.std(dif(target, predicted)) / predicted, color='g', alpha=0.5)

    if limit:
        plt.xlim(lim_l, lim_r)
        plt.ylim(0, lim_y)

    plt.legend()
    plt.show()
    # plt.savefig("stdXenergy_%d_%d.jpg" % (lim_l, lim_r))
    
    
def binning(nbins, label, pred):
    '''
    Divides the data into n bins containing energy ranges of the same size.
    :parameter nbins: number of bins.
    :type nbins: int
    :parameter label: array of energy targets (true value label).
    :type label: numpy.ndarray
    :parameter pred: array of predictions from testing.
    :type pred: numpy.ndarray
    :return: arrays of means, relative means, standard deviations, relative standard deviations, and size of the bins.
    :rtype: array, array, array, array, array.
    '''

    if __package__ is None:
        sys.path.append(os.path.realpath("/data/shared/Software/CMS_Deep_Learning"))
    
    from CMS_Deep_Learning.io import gen_from_data, retrieve_data
    from CMS_Deep_Learning.postprocessing.metrics import distribute_to_bins
    
    out, x, y = distribute_to_bins(label, [label, pred], nb_bins=nbins, equalBins=True)
    iSize = 500/nbins
    
    means = []
    rMeans = [] # normalized means
    stds = []
    rStds = [] # normalized standard deviations
    sizes = [] # number of events in the bins
    
    for i in range(0, nbins):
        sizes.append(len(x[i]))
        plotPredictedXTarget(x[i], y[i], i*iSize, (i+1)*iSize)
        histEDif(x[i], y[i], nbins=200, lim=20, lim_l=i*iSize, lim_r=(i+1)*iSize)
        histRelDif(x[i], y[i], nbins=150, lim=0.15, lim_l=i*iSize, lim_r=(i+1)*iSize)
        
        difference = dif(x[i], y[i])
        relDiff = rDif(x[i], y[i])
        
        mean = np.mean(difference)
        means.append(mean)
        
        rMean = np.mean(relDiff)
        rMeans.append(rMean)
        
        std = np.std(difference)
        stds.append(std)
        
        rStd = np.std(relDiff)
        rStds.append(rStd)
        
    
    return x, y, means, rMeans, stds, rStds, sizes


def plotN(input, stds, sizes, what):
    '''
    Plots the means or stds (normalized or not) for the bins of energy.
    :param input: array containing the data to be plotted (means, rMeans, stds or rStds).
    :type input: array
    :param stds: stds array to calculate the error.
    :type stds: array
    :param sizes: array containing the number of samples in each bin, to calculate the error.
    :type sizes: array
    :param what: what is plotted (means, rMeans, stds or rStds), for the legend.
    :type what: str
    '''
    plt.figure(figsize=(5, 5))
    plt.xlabel("Energy bin", size=18)

    n = len(input)
    # print n
    iSize = 500 / n

    if what == "means":
        for i in range(0, n):
            error = stds[i] / np.sqrt(sizes[i])
            plt.scatter(i, input[i], color=tuple(np.random.random(3)), alpha=0.5,
                        label='%d to %d GeV' % (i * iSize, (i + 1) * iSize))
            plt.errorbar(i, input[i], yerr=error)

        plt.ylabel("$\mu_{\Delta E}$ (GeV)", size=19)
        plt.title("Means", size=18)


    elif what == "stds":
        for i in range(0, n):
            plt.scatter(i, input[i], color=tuple(np.random.random(3)), alpha=0.5,
                        label='%d to %d GeV' % (i * iSize, (i + 1) * iSize))

        plt.ylabel("$\sigma_{\Delta E}$ (GeV)", size=19)
        plt.title("Standard deviations", size=18)


    elif what == "rMeans":
        for i in range(0, n):
            error = stds[i] / np.sqrt(sizes[i])
            plt.scatter(i, input[i], color=tuple(np.random.random(3)), alpha=0.5,
                        label='%d to %d GeV' % (i * iSize, (i + 1) * iSize))
            plt.errorbar(i, input[i], yerr=error)

        plt.ylabel(r"$\mu_{\frac{\Delta E}{E}}$ (%)", size=19)
        plt.title("Relative means", size=18)

    elif what == "rStds":
        for i in range(0, n):
            plt.scatter(i, input[i], color=tuple(np.random.random(3)), alpha=0.5,
                        label='%d to %d GeV' % (i * iSize, (i + 1) * iSize))

        plt.ylabel(r"$\sigma_{\frac{\Delta E}{E}}$ (%)", size=19)
        plt.title("Relative standard deviations", size=18)

    else:
        raise ValueError("The second parameter should be 'means', 'stds', 'rMeans' or 'rStds'. ")

    plt.xlim(-0.9, 10)
    plt.legend(loc='best', bbox_to_anchor=(1.52, 0.9))
    plt.show()
    # plt.savefig("means.jpg")