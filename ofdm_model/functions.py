import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.ticker import NullFormatter


def qpsk_modulator(x):
    output = []
    for i in range(len(x)/2):
        re = x[2*i]
        im = x[2*i+1]
        remod = -1 if re == 0 else 1
        immod = -1j if im == 0 else 1j
        output.append(remod + immod)
    return output

def qpsk_demodulator(x):
    output = []
    for i in x:
        redemod = 0 if np.sign(i.real) < 0 else 1
        imdemod = 0 if np.sign(i.imag) < 0 else 1
        output.append(redemod)
        output.append(imdemod)
    return output

def scatterplot(real,imag):
    # Definitions for the plot axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65

    rect_scatter = [left, bottom, width, height]
    scatterplot = plt.axes(rect_scatter)
    # draw axes at origin
    scatterplot.axhline(0, color='black')
    scatterplot.axvline(0, color='black')

    scatterplot.scatter(real, imag)
    plt.show()

def constellation(array,SNR,BER):

    array = array.T
    # definitions for the plot axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular figure
    plt.figure(1, figsize=(8, 8))

    # set up plots
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no axis labels for box plots
    axHistx.xaxis.set_major_formatter(NullFormatter())
    axHisty.yaxis.set_major_formatter(NullFormatter())

    # scatter plot:
    axScatter.scatter(array.real, array.imag)

    # draw axes at origin
    axScatter.axhline(0, color='black')
    axScatter.axvline(0, color='black')

    # add title (at x-axis) to scatter plot
    # title = 'Zero noise'
    EbN0 = 10 ** (SNR / 10.)
    ThBER = berqpsk(EbN0)
    title = 'SNR = %sdB with a BER of %.5g (Theoretical BER = %.5g)' % (SNR, float(BER), ThBER)
    axScatter.xaxis.set_label_text(title)

    # now determine nice limits by hand:
    binwidth = 0.5  # width of histrogram 'bins'
    xymax = np.max([np.max(np.fabs(array.real)),
                    np.max(np.fabs(array.imag))])  # find abs max symbol value; nominally 1
    lim = (int(xymax / binwidth) + 1) * binwidth  # create limit that is one 'binwidth' greater than 'xymax'

    axScatter.set_xlim((-lim, lim))  # set the data limits for the xaxis -- autoscale
    axScatter.set_ylim((-lim, lim))  # set the data limits for the yaxis -- autoscale

    bins = np.arange(-lim, lim + binwidth, binwidth)  # create bins 'binwidth' apart between -lin and +lim -- autoscale
    axHistx.hist(array.real, bins=bins)  # plot a histogram - xaxis are real values
    axHisty.hist(array.imag, bins=bins, orientation='horizontal')  # plot a histogram - yaxis are imaginary values

    axHistx.set_xlim(axScatter.get_xlim())  # set histogram axes to match scatter plot axes limits
    axHisty.set_ylim(axScatter.get_ylim())  # set histogram axes to match scatter plot axes limits
    #plt.ion()
    plt.show()
    #plt.pause(0.001)

def qfunc(x):
    return 0.5*math.erfc(x/(np.sqrt(2)))

def berqpsk(EbN0):
    return qfunc(np.sqrt(2*(EbN0)))