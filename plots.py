'''
plots.py
Contains custom utilities for plotting and displaying tested data.
Author: Vitoria Barin Pacela
e-mail: vitoria.barimpacela@helsinki.fi
'''
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os, sys
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
#import matplotlib.mlab as mlab
#from scipy.stats import norm

if __package__ is None:
    sys.path.append(os.path.realpath("/data/shared/Software/CMS_Deep_Learning"))

from CMS_Deep_Learning.io import gen_from_data, retrieve_data
from CMS_Deep_Learning.postprocessing.metrics import distribute_to_bins

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


def plotPredictedXTarget(target, predicted, lim_l=0, lim_r=550, particle=""):
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
    :parameter particle: name of the particle in the dataset, for the title.
    :type particle: str
    '''

    plt.figure(figsize=(5, 5))
    plt.xlabel("True energy (GeV)")
    plt.ylabel("Predicted energy (GeV)")
    plt.title(particle)
    #plt.title(u"%s Predicted X true energy \n Energies between %d and %d GeV" % (particle, lim_l, lim_r))

    plt.scatter(target, predicted, color='g', alpha=0.5)

    plt.xlim(lim_l, lim_r)
    plt.ylim(lim_l, lim_r)

    plt.legend()
    plt.show()


def PredictedTarget(target, predicted, lim_l=0, lim_r=550, particle="", nbins=200):
    '''
    Plots the predicted energy against the target energy as a 2D histogram.
    :parameter target: array of energy targets (true value label).
    :type target: numpy.ndarray
    :parameter predicted: array of predictions from testing.
    :type predicted: numpy.ndarray
    :parameter lim_l: minimum value for both the x and y axes.
    :type lim_l: float
    :parameter lim_r: maximum value for both the x and y axes.
    :type lim_r: float
    :parameter particle: name of the particle in the dataset, for the title.
    :type particle: str
    '''


    plt.figure(figsize=(5, 5))
    plt.xlabel("True energy (GeV)")
    plt.ylabel("Predicted energy (GeV)")
    plt.title(particle)
    # plt.title(u"%s Predicted X true energy \n Energies between %d and %d GeV" % (particle, lim_l, lim_r))

    # plt.scatter(target, predicted, color='g', alpha=0.5, cmap=cm)
    plt.hist2d(target, predicted, bins=nbins, norm=LogNorm(), cmap="cool")

    plt.xlim(lim_l, lim_r)
    plt.ylim(lim_l, lim_r)

    plt.legend()
    plt.colorbar()
    plt.show()


def histEDif(target, pred, nbins=1500, lim=25, lim_l=0, lim_r=550, particle=""):
    '''
    Plots histogram of the difference between the target energy and the predicted energy (GeV).
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
    :parameter particle: name of the particle in the dataset, for the title.
    :type particle: str
    '''
    difference = dif(target, pred)

    # the histogram of the data

    mean = np.mean(difference)
    # print(mean)

    std = np.std(difference)  # standard deviation
    # print(std)

    error = std / np.sqrt(len(target))
    # labels=['Mean: %.2f' % mean, 'Standard deviation: %.2f' % std]

    n, bins, patches = plt.hist(difference, nbins, normed=1, facecolor='green', alpha=0.75)

    # plt.text(10, 0.025, 'Mean: %.2f $\pm$ %.2f \nStd. dev.: %.2f' % (mean, error, std))

    # plt.xlabel('Difference between true and predicted energy (GeV)')
    plt.xlabel(r'$E_{true} - E_{pred}$ (GeV)', size=16)
    plt.ylabel('Probability', size=16)
    # plt.title("Energy difference \n Energies between %d and %d GeV" % (lim_l, lim_r), size=16)
    plt.title(u"%s Energy difference" % particle, size=16)

    plt.xlim(-lim, lim)

    # text has to be defined after setting the limits
    plt.text(0.3 * max(plt.xlim()), 0.6 * max(plt.ylim()),
             'Mean: %.2f $\pm$ %.2f \nStd. dev.: %.2f' % (mean, error, std))
    # print(max(plt.xlim()))
    # print(type(max(plt.xlim())))
    # plt.axis('tight')
    plt.show()
    # plt.savefig("histDif_%d_%d.jpg" % (lim_l, lim_r))


def resolution(true, pred, particle=""):
    '''
    NOT USEFUL.
    Plots the energy resolution of the calorimeter for the data used.
    :parameter true: array of energy targets (true value label).
    :type true: numpy.ndarray
    :parameter pred: array of predictions from testing.
    :type pred: numpy.ndarray
    :parameter particle: name of the particle in the dataset, for the title.
    :type particle: str
    '''
    plt.rcParams['agg.path.chunksize'] = 10000
    from scipy.optimize import curve_fit

    res = np.std(dif(true, pred)) / true

    # energy resolution of the calorimeter
    def func(res, a, b, c):
        # equation to be fit in the data
        return a / np.sqrt(res) + b + c / res

    popt, pcov = curve_fit(func, true, res)
    print(popt)
    print(pcov)

    # sorting the data so that it can be plot
    import itertools
    y = func(true, *popt)
    lists = sorted(itertools.izip(*[true, y]))
    new_x, new_y = list(itertools.izip(*lists))

    fit = r'$\frac{\sigma(\Delta E)}{E_{t}} = \frac{%.2e}{\sqrt{E_{t}}} + %.2e + \frac{%.2e}{E_{t}}$' % (
    popt[0], popt[1], popt[2])

    plt.plot(new_x, new_y, 'r', label=fit)

    plt.scatter(true, res)

    plt.xlim(0, 500)
    plt.title("%s Energy Resolution" % particle)
    plt.xlabel("True energy (GeV)")
    plt.ylabel(r"$\frac{\sigma(\Delta E)}{E_t}$", size=18)
    plt.legend(prop={'size': 15})


def histRelDif(target, pred, nbins=550, lim=20, lim_l=0, lim_r=550, particle=""):
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
    :parameter particle: name of the particle in the dataset, for the title.
    :type particle: str
    '''
    difference = rDif(target, pred)

    # the histogram of the data

    mean = np.mean(difference)
    # print(mean)
    std = np.std(difference)  # standard deviation
    # print(std)
    error = std / np.sqrt(len(target))
    # print(error)

    n, bins, patches = plt.hist(difference, nbins, normed=1, facecolor='green', alpha=0.75)

    plt.xlim(-lim, lim)
    plt.text(0.3 * max(plt.xlim()), 0.6 * max(plt.ylim()),
             'Mean: %.3f $\pm$ %.3f \nStd. dev.: %.2f' % (mean, error, std))

    plt.xlabel(r'$\frac{(E_{true} - E_{pred})}{E_{true}}$ (%)', size=18)
    plt.ylabel('Probability', size=16)
    plt.title("%s Relative energy difference" % particle)

    plt.show()
    # plt.savefig("histRDif_%d_%d.jpg" % (lim_l, lim_r))


def RelTarget(target, pred, particle="", nbins=200):
    '''
    Plots the relative energy difference against the target energy, as a 2D histogram.
    :parameter target: array of energy targets (true value label).
    :type target: numpy.ndarray
    :parameter pred: array of the energy predictions.
    :type pred: numpy.ndarray.
    :parameter particle: name of the particle in the dataset, for the title.
    :type particle: str
    '''

    plt.figure(figsize=(5, 5))
    plt.xlabel("True energy (GeV)")
    #plt.ylabel("Relative energy difference (%)")
    plt.ylabel(r'$\frac{(E_{true} - E_{pred})}{E_{true}}$ (%)', size=18)
    plt.title("%s \n Relative energy difference X True energy" % particle)

    rDifference = rDif(target, pred)

    plt.hist2d(target, rDifference, bins=nbins, norm=LogNorm(), cmap="cool")

    # plt.xlim(lim_l, lim_r)
    plt.ylim(-100, 100)  # there seems to be a bug in pyplot that creates a bunch of white space in the figure
    plt.colorbar()
    # plt.axis()

    plt.show()
    # plt.savefig("relXtarget_%d_%d.jpg" % (lim_l, lim_r))


def plotRelXTarget(target, relative, lim_l=0, lim_r=500, particle=""):
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
    :parameter particle: name of the particle in the dataset, for the title.
    :type particle: str
    '''

    plt.figure(figsize=(5, 5))
    plt.xlabel("Target energy (GeV)")
    plt.ylabel("Relative energy difference (%)")
    plt.title("%s \n Relative energy difference X True energy" % particle)

    plt.scatter(target, relative, color='g', alpha=0.5)

    plt.xlim(lim_l, lim_r)
    plt.ylim(-100, 100)
    plt.legend()
    plt.show()
    # plt.savefig("relXtarget_%d_%d.jpg" % (lim_l, lim_r))


def plotMeanXEnergy(target, predicted, lim_y=0.14, lim_l=0, lim_r=520, limit=False):
    '''
    Not so useful.
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
    Not so useful.

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


def binning(nbins, label, pred, plot=False):
    '''
    Divides the data into n bins containing energy ranges of the same size.
    :parameter nbins: number of bins.
    :type nbins: int
    :parameter label: array of energy targets (true value label).
    :type label: numpy.ndarray
    :parameter pred: array of predictions from testing.
    :type pred: numpy.ndarray
    :parameter plot: whether to plot Predicted X Target and histogram for each energy bin.
    :type plot: bool
    :return: arrays of means, relative means, standard deviations, relative standard deviations, size of the bins, and energy resolution.
    :rtype: array, array, array, array, array, array.
    '''

    #if __package__ is None:
    #    sys.path.append(os.path.realpath("/data/shared/Software/CMS_Deep_Learning"))

    #from CMS_Deep_Learning.io import gen_from_data, retrieve_data
    #from CMS_Deep_Learning.postprocessing.metrics import distribute_to_bins

    out, x, y = distribute_to_bins(label, [label, pred], nb_bins=nbins, equalBins=True)
    # x -> true energy
    # y -> pred energy
    iSize = 500 / nbins

    means = []
    rMeans = []  # normalized means
    stds = []
    rStds = []  # normalized standard deviations
    sizes = []  # number of events in the bins
    res= [] # calorimeter energy resolution

    for i in range(0, nbins):
        sizes.append(len(x[i]))

        if (plot == True):
            #plotPredictedXTarget(x[i], y[i], i * iSize, (i + 1) * iSize)
            PredictedTarget(x[i], y[i], i * iSize, (i + 1) * iSize)
            # histEDif(x[i], y[i], nbins=200, lim=20, lim_l=i*iSize, lim_r=(i+1)*iSize)
            histRelDif(x[i], y[i], nbins=150, lim=15, lim_l=i*iSize, lim_r=(i+1)*iSize)

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

        eRes = std / np.mean(x[i])
        res.append(eRes)

    return x, y, means, rMeans, stds, rStds, sizes, res


def plotN(inp, stds, sizes, what, particle=""):
    '''
    TOO LONG AND COMPLICATED. CHECK METHODS BELOW.
    Plots the means or stds (normalized or not) for the bins of energy.
    :param inp: array containing the data to be plotted (means, rMeans, stds or rStds).
    :type inp: array
    :param stds: stds array to calculate the error.
    :type stds: array
    :param sizes: array containing the number of samples in each bin, to calculate the error.
    :type sizes: array
    :param what: what is plotted (means, rMeans, stds or rStds), for the legend.
    :type what: str
    :parameter particle: name of the particle in the dataset, for the title.
    :type particle: str
    '''
    plt.figure(figsize=(5, 5))
    plt.xlabel("Energy", size=16)

    n = len(inp)
    # print(n)
    iSize = 500 / n

    if what == "means":
        for i in range(0, n):
            x_axis = (i * iSize + (i + 1) * iSize) / 2
            error = stds[i] / np.sqrt(sizes[i])
            plt.scatter(x_axis, inp[i], color='purple', alpha=0.5
                        # , label='%d to %d GeV' % (i * iSize, (i + 1) * iSize)
                        )
            plt.errorbar(x_axis, inp[i], yerr=error, color='black')

        plt.ylabel("$\mu_{\Delta E}$ (GeV)", size=19)
        plt.title("%s Means" % particle, size=16)

    elif what == "rMeans":
        for i in range(0, n):
            energy = (i * iSize + (i + 1) * iSize) / 2
            error = stds[i] / np.sqrt(sizes[i])
            plt.scatter(energy, inp[i], color='pink', alpha=0.8
                        # , label='%d to %d GeV' % (i * iSize, (i + 1) * iSize)
                        )
            plt.errorbar(energy, inp[i], yerr=error, color='purple')

        plt.ylabel(r"$\mu_{\frac{\Delta E}{E}}$ (%)", size=19)
        plt.title("%s Relative means" % particle, size=16)

    elif what == "stds":
        for i in range(0, n):
            energy = (i * iSize + (i + 1) * iSize) / 2
            plt.scatter(energy, inp[i], color='blue', alpha=0.5
                        # , label='%d to %d GeV' % (i * iSize, (i + 1) * iSize)
                        )

        plt.ylabel("$\sigma_{\Delta E}$ (GeV)", size=19)
        plt.title("%s Standard deviations" % particle, size=16)

    elif what == "rStds":
        for i in range(0, n):
            energy = (i * iSize + (i + 1) * iSize) / 2
            plt.scatter(energy, inp[i], color='orange', alpha=0.5
                        # , label='%d to %d GeV' % (i * iSize, (i + 1) * iSize)
                        )

        plt.ylabel(r"$\sigma_{\frac{\Delta E}{E}}$ (%)", size=19)
        plt.title("%s Relative standard deviations" % particle, size=16)

    elif what == "res":
        for i in range(0, n):
            energy = (i * iSize + (i + 1) * iSize) / 2
            plt.scatter(energy, inp[i], color='blue', alpha=0.5
                        # , label='%d to %d GeV' % (i * iSize, (i + 1) * iSize)
                        )

        plt.ylabel(r"$\frac{\sigma({\Delta E})}{E_{mean}}$ (GeV)", size=19)
        plt.title("%s Energy resolution" % particle, size=16)

    else:
        raise ValueError("'what' should be 'means', 'stds', 'rMeans', 'rStds', or 'res'. ")

    plt.xlim(0, 500)
    plt.legend(loc='best', bbox_to_anchor=(1.52, 0.9))
    plt.show()
    # plt.savefig("means.jpg")


def means(means, stds, sizes, particle =""):
    '''
    Plots the absolute mean for each bin of energy.
    :param means: array containing the means to be plotted.
    :type means: array
    :param stds: stds array to calculate the error.
    :type stds: array
    :param sizes: array containing the number of samples in each bin, to calculate the error.
    :type sizes: array
    :parameter particle: name of the particle in the dataset, for the title.
    :type particle: str
    '''
    plt.figure(figsize=(5, 5))

    n = len(means)
    # print(n)
    iSize = 500 / n

    for i in range(0, n):
        x_axis = (i * iSize + (i + 1) * iSize) / 2
        error = stds[i] / np.sqrt(sizes[i])
        plt.scatter(x_axis, means[i], color='purple', alpha=0.5
                    # , label='%d to %d GeV' % (i * iSize, (i + 1) * iSize)
                    )
        plt.errorbar(x_axis, means[i], yerr=error, color='black')

    plt.xlabel("Energy", size=16)
    plt.ylabel("$\mu(\Delta E)$ (GeV)", size=19)
    plt.title("%s Means" % particle, size=16)
    plt.xlim(0, 500)
    #plt.legend(loc='best', bbox_to_anchor=(1.52, 0.9))
    plt.show()
    # plt.savefig("means.jpg")


def rMeans(rMeans, stds, sizes, particle =""):
    '''
    Plots the relative mean for each bin of energy.
    :param rMeans: array containing the relative means.
    :type rMeans: array
    :param stds: stds array to calculate the error.
    :type stds: array
    :param sizes: array containing the number of samples in each bin, to calculate the error.
    :type sizes: array
    :parameter particle: name of the particle in the dataset, for the title.
    :type particle: str
    '''
    plt.figure(figsize=(5, 5))

    n = len(rMeans)
    # print(n)
    iSize = 500 / n

    for i in range(0, n):
        energy = (i * iSize + (i + 1) * iSize) / 2
        error = stds[i] / np.sqrt(sizes[i])
        plt.scatter(energy, rMeans[i], color='pink', alpha=0.8
                    # , label='%d to %d GeV' % (i * iSize, (i + 1) * iSize)
                    )
        plt.errorbar(energy, rMeans[i], yerr=error, color='purple')

    plt.xlabel("Energy", size=16)
    plt.ylabel(r"$\mu(\frac{\Delta E}{E_{true}})$ (%)", size=19)
    plt.title("%s Relative means" % particle, size=16)
    plt.xlim(0, 500)
    #plt.legend(loc='best', bbox_to_anchor=(1.52, 0.9))
    plt.show()
    # plt.savefig("rMeans.jpg")


def stds(stds, particle =""):
    '''
    Plots the absolute standard deviation for each bin of energy.
    :param stds: array containing the standard deviations.
    :type stds: array
    :parameter particle: name of the particle in the dataset, for the title.
    :type particle: str
    '''

    n = len(stds)
    # print(n)
    iSize = 500 / n

    plt.figure(figsize=(5, 5))

    for i in range(0, n):
        energy = (i * iSize + (i + 1) * iSize) / 2
        plt.scatter(energy, stds[i], color='blue', alpha=0.5
                    # , label='%d to %d GeV' % (i * iSize, (i + 1) * iSize)
                    )

    plt.xlabel("Energy", size=16)
    plt.ylabel("$\sigma(\Delta E)$ (GeV)", size=19)
    plt.title("%s Standard deviations" % particle, size=16)
    plt.xlim(0, 500)
    #plt.legend(loc='best', bbox_to_anchor=(1.52, 0.9))
    plt.show()
    # plt.savefig("stds.jpg")


def rStds(rStds, particle =""):
    '''
    Plots the relative standard deviation for each bin of energy.
    :param rStds: array containing the relative standard deviations.
    :type rStds: array
    :parameter particle: name of the particle in the dataset, for the title.
    :type particle: str
    '''
    plt.figure(figsize=(5, 5))

    n = len(rStds)
    # print(n)
    iSize = 500 / n

    for i in range(0, n):
        energy = (i * iSize + (i + 1) * iSize) / 2
        plt.scatter(energy, rStds[i], color='orange', alpha=0.5
                    # , label='%d to %d GeV' % (i * iSize, (i + 1) * iSize)
                    )

    plt.xlabel("Energy", size=16)
    plt.ylabel(r"$\sigma(\frac{\Delta E}{E_{true}})$ (%)", size=19)
    plt.title("%s Relative standard deviations" % particle, size=16)
    plt.xlim(0, 500)
    #plt.legend(loc='best', bbox_to_anchor=(1.52, 0.9))
    plt.show()
    # plt.savefig("rStds.jpg")


def res(res, particle="", verbose=False):
    '''
    Plots the energy resolution of the calorimeter and fits its equation.
    :param res: array containing the standard deviation divided by the mean of each bin.
    :type res: array
    :parameter particle: name of the particle in the dataset, for the title.
    :type particle: str
    :parameter verbose: whether to print a, b and c for the fit.
    :type verbose: bool
    '''
    plt.figure(figsize=(5, 5))

    n = len(res)
    # print(n)
    iSize = 500 / n

    energies = []

    for i in range(0, n):
        energy = (i * iSize + (i + 1) * iSize) / 2
        energies.append(energy)

        plt.scatter(energy, res[i], color='blue', alpha=0.5
                    # , label='%d to %d GeV' % (i * iSize, (i + 1) * iSize)
                    )

    ####### energy resolution of the calorimeter ############
    plt.rcParams['agg.path.chunksize'] = 10000
    #from scipy.optimize import curve_fit

    def func(E, a, b, c):
        # equation to be fit in the data
        return a / np.sqrt(E) + b + c / E

    popt, pcov = curve_fit(func, energies, res)
    if (verbose == True):
        print(popt)
        # print(pcov)

    y = func(energies, *popt)

    fit = r'$\frac{\sigma(\Delta E)}{E_{t}} = \frac{%.2e}{\sqrt{E_{t}}} + %.2e + \frac{%.2e}{E_{t}}$' % (
        popt[0], popt[1], popt[2])

    plt.plot(energies, y, 'r', label=fit)

    # sorting the data so that it can be plot
    # import itertools
    # lists = sorted(itertools.izip(*[n, y]))
    # new_x, new_y = list(itertools.izip(*lists))

    # plt.plot(new_x, new_y, 'r', label=fit)
    #########################################################

    plt.xlabel("Energy", size=16)
    plt.ylabel(r"$\frac{\sigma({\Delta E})}{E_{true}}$ (GeV)", size=19)
    plt.title("%s Energy resolution" % particle, size=16)
    plt.xlim(0, 500)
    plt.legend(loc='best', bbox_to_anchor=(1.52, 0.9))
    plt.show()
    # plt.savefig("res.jpg")


def plotBins(nbins, true, pred, particle=""):
    '''
    Distribute the data to bins and plots the means, relative means, standard deviations, relative standard deviations and the energy resolution.
    The energies are divided into n bins of the same size.
    :parameter nbins: number of bins.
    :type nbins: int
    :parameter true: array of energy targets (true value label).
    :type true: numpy.ndarray
    :parameter pred: array of predictions from testing.
    :type pred: numpy.ndarray
    '''
    x_ar, y_ar, means_ar, rMeans_ar, stds_ar, rStds_ar, sizes_ar, res_ar = binning(nbins, true, pred)
    means(means_ar, stds_ar, sizes_ar, particle=particle)
    rMeans(rMeans_ar, stds_ar, sizes_ar, particle=particle)
    stds(stds_ar, particle=particle)
    rStds(rStds_ar, particle=particle)
    res(res_ar, particle=particle)
    res(rStds_ar, particle=particle)


def plot_all(true, pred, particle="", nbins=10):
    '''
    Creates all the relevant plots for the analysis.
    :parameter true: array of energy targets (true value label).
    :type true: numpy.ndarray
    :parameter pred: array of predictions from testing.
    :type pred: numpy.ndarray
    :parameter particle: particle type (electron, photon, charged pion, neutral pion)
    :type particle: str
    :parameter nbins: number of bins.
    :type nbins: int
    '''
    PredictedTarget(true, pred, particle=particle)
    histEDif(true, pred, particle=particle)
    histRelDif(true, pred, particle=particle)
    RelTarget(true, pred, particle=particle)
    plotBins(nbins, true, pred, particle=particle)


def plotSumXTarget(target, inSum):
    '''
    Plots the sum of energies against the true energy.
    :type target: numpy.ndarray.
    :param target: array of true energy values.
    :type inSum: numpy.ndarray.
    :param inSum: array of the input energy sums.
    '''

    plt.figure(figsize=(5, 5))
    plt.xlabel("True energy (GeV)")
    plt.ylabel("Summed energy (GeV)")
    plt.title("Sum X True energy")

    plt.scatter(target, inSum, color='g', alpha=0.5)

    plt.xlim(0, 500)
    #plt.ylim(0, 500)

    plt.legend()
    plt.show()


def SumTarget(target, inSum, particle =""):
    '''
    Plots the sum of energies against the true energy as a 2D histogram.
    :type target: numpy.ndarray.
    :parameter target: array of true energy values.
    :type inSum: numpy.ndarray.
    :parameter inSum: array of the input energy sums.
    :type particle: str
    :parameter particle: name of the particle
    '''
    plt.figure(figsize=(5, 5))
    plt.xlabel("True energy (GeV)")
    plt.ylabel("Summed energy (GeV)")
    plt.title(particle + "Sum X True energy")

    plt.hist2d(target, inSum, bins=200, norm=LogNorm(), cmap="cool")

    plt.xlim(0, 500)
    #plt.ylim(0, 500)

    plt.legend()
    plt.colorbar()
    plt.show()


def saveTruePred(name, true, pred):
    '''
    Saves true energy and prediction energy arrays into an HDF5 file.
    :parameter name: name of the file to be saved.
    :type name: str.
    :parameter true: array of energy targets (true value label).
    :type true: numpy.ndarray
    :parameter pred: array of predictions from testing.
    :type pred: numpy.ndarray
    '''
    true_pred = np.array([true, pred])
    # to fix and implement: if file is already open, continue
    new_file = h5py.File(name + "TruePred.h5", 'w')
    ds = new_file.create_dataset("true_pred", data=true_pred)
    new_file.close()


def true_pred_from_HDF5(file_name):
    '''
    Loads true and predicted energy arrays from HDF5 file.
    :parameter file_name: name of the HDF5 file containing true and pred arrays.
    :type file_name: str
    :return: true and pred arrays.
    :rtype: numpy.ndarray, numpy.ndarray
    '''
    f = h5py.File(file_name + ".h5", 'r')
    true, pred = f["true_pred"][0], f["true_pred"][1]
    return true, pred