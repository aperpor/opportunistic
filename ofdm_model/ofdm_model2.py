import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from scipy import signal
import timeit
import functions as fn
import time
import sys

####################################
#            Variables             #
####################################
#Whether to plot things or not (Speeds it up)
plot = True
#Upsampling constant
k = 20
#SNR
SNR = 0
# Es/N0
EsN0 = 10 ** (SNR / 10.)
EbN0 = EsN0 / 2
# N0 (The dot is to force the division to be a float
N0 = 2. / EsN0
# Noise scale
scale = np.sqrt(N0)
#Sampling Frequency
fs = 4e6

#Number of channels
M = 100
#Number of bursts
block = 1024

errors = 0
totalsent = 0

count = 0
nperiod = 20

plt.ion()

if plot:
    win = pg.GraphicsWindow(title="OFDM Bandpass Model")
    p1 = win.addPlot(title="IFFT")
    p1.setLogMode(False,True)
    #p1.showGrid(1,1,1)
    p2 = win.addPlot(title="Modulated & Upsampled")
    p2.setLogMode(False,True)
    p3 = win.addPlot(title="Real part plus AWGN")
    p3.setLogMode(False,True)
    p4 = win.addPlot(title="Demodulated & Downsampled")
    #p4.setLogMode(False,True)
    win.nextRow()
    p5 = win.addPlot(title="Receiver constellation")

    count = 0
    x2m = []
    x3m = []

while (errors < 100):

    ####################################
    #           Transmission           #
    ####################################

    # We generate M * block random bits
    source = np.random.choice([0,1],M*block,p=[0.5,0.5])
    # We reshape the input to obtain the parallel streams
    parallel = np.frombuffer(source).reshape(block, M).T
    # Constellation mapping of each stream using QPSK
    mappings = []
    for stream in parallel:
        mappings.append(fn.qpsk_modulator(stream))
    # Array of ready-to-send M-sized vectors
    outputs = np.asarray(mappings).T

    # IFFT for each M-sized vector
    tx = []
    for out in outputs:
        tx.append(np.fft.ifft(out, norm="ortho"))
    tx = np.asarray(tx)
    '''New part'''
    #We upsample by a k factor
    rx = []
    n = np.arange(len(tx[0])*k)
    mod = np.e ** (1j * np.pi * 0.5 * n)
    demod = np.e ** (-1j * np.pi * 0.5 * n)
    for bl in tx:
        '''Plot the original IFFT output'''
        if plot:
            f1, x1 = signal.periodogram(bl, fs/k, 'flattop')
            p1.plot(np.fft.fftshift(f1), np.fft.fftshift(x1), clear=True)

        '''Upsampling by k'''
        aux = signal.resample(bl,len(bl)*k) / np.sqrt(k)
        #print abs(sum(aux) / len(aux))

        '''Modulate at carrier frequency'''
        upmod = np.asarray(aux * mod)

        '''Plot the upsampled and modulated signal'''
        if plot:
            f2, x2 = signal.periodogram(upmod, fs, 'flattop', scaling="spectrum")
            count += 1
            if count <nperiod:
                if len(x2m) == 0:
                    x2m = x2
                else:
                    x2m += x2
            else:
                x2m += x2
                x2m = np.asarray(x2m) / nperiod
                p2.plot(f2,x2m, clear=True)
                x2m = []
            #p2.plot(f2,x2, clear = True)

        '''Add AWGN'''
        noise = np.random.normal(scale=scale, size=len(upmod.real))
        # noise = np.fft.ifft(noise)
        received = upmod.real + noise

        '''Plot the received signal'''
        if plot:
            f3, x3 = signal.periodogram(received, fs, 'flattop', scaling="spectrum")
            if count <nperiod:
                if len(x3m) == 0:
                    x3m = x3
                else:
                    x3m += x3
            else:
                x3m += x3
                x3m = np.asarray(x3m) / nperiod
                p3.plot(f3,x3m, clear=True)
                x3m = []
                count = 0

        '''Demodulate'''
        aux = np.asarray(received * demod)

        b, a = signal.butter(13, 300e3/(fs/2))
        w, h = signal.freqz(b,a)
        #plt.plot(w,10*np.log(np.abs(h)))
        #plt.semilogx()
        #plt.pause(0.05)
        demodfiltered = signal.filtfilt(b, a, aux)

        '''Downsampling by k'''
        downdemod = signal.resample(demodfiltered,len(aux)/k)

        if plot:
            f4, x4 = signal.periodogram(aux, fs, 'flattop', scaling="spectrum")
            f5, x5 = signal.periodogram(demodfiltered, fs, 'flattop', scaling="spectrum")
            p4.plot(np.fft.fftshift(f4), np.fft.fftshift(x4), pen=(255,0,0), name="Red Curve", clear=True)
            p4.plot(np.fft.fftshift(f5), np.fft.fftshift(x5), pen=(0,255,0), name="Green Curve")
        rx.append(downdemod)
        if plot:
            pg.QtGui.QApplication.processEvents()

    #print "Finished transmission."e
    # FFT for each M-sized vector
    y = []
    for received in rx:
        y.append(np.fft.fft(received, norm="ortho"))
    y = np.asarray(y)

    '''Plot constellation'''
    '''toplot = y.reshape(y.size)
    p5.plot(toplot.real, toplot.imag, pen=None, symbol='t', symbolPen=None, symbolSize=10, symbolBrush=(100, 100, 255, 50))
    pg.QtGui.QApplication.processEvents()'''

    #y = np.asarray(rx)
    # Demodulation
    demod = []
    for vec in y.T:
        demod.append(fn.qpsk_demodulator(vec))

    demod = np.asarray(demod)

    # Parallel to Serial
    output = np.reshape(demod.T, M * block)
    # print output
    err = np.where(source != output)
    errors += len(err[0])
    totalsent += M * block
    sys.stdout.write('\r' + str(errors) + " errors")
BER = errors / float(totalsent)
print "\nSimulation BER: " + str(BER)
print "Theoretical BER: " + str(fn.berqpsk(EsN0))

while True:
    pg.QtGui.QApplication.processEvents()
