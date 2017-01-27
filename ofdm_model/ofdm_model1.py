import numpy as np
import matplotlib.pyplot as plt
import functions as fn
from scipy.interpolate import spline

####################################
#            Variables             #
####################################

#Number of channels
M = 100
#Size of each OFDM stream
block = 1024
#SNR
SNR=20

#Length of the cyclic prefix
cp_len = np.math.floor(0.1*block)


def ofdmqpsksimulation(SNR, plots=False):
    #Initial setup
    # Eb/N0
    EbN0 = 10 ** (SNR / 10.)
    # N0 (The dot is to force the division to be a float
    N0 = 1. / EbN0
    # Noise scale
    scale = np.sqrt(N0 / 2)

    ####################################
    #           Transmission           #
    ####################################

    #We generate M * block random bits
    source = np.round(np.random.rand(1,int(M*block)))[0]
    #print "This is the original source in bits"
    #print source

    #We reshape the input to obtain the parallel streams
    parallel = np.frombuffer(source).reshape(block,M).T
    #print "\nDividing the stream in M channels with a serial to parallel"
    #print parallel

    #Constellation mapping of each stream using QPSK
    mappings = []
    for stream in parallel:
        mappings.append(fn.qpsk_modulator(stream))
    #print "\nThis is the mapping of each stream"
    #print np.asarray(mappings)

    #Array of ready-to-send M-sized vectors
    outputs = np.asarray(mappings).T
    #print "\nEach column of the previous matrix goes into the IFFT block. Here they are represented as row-arrays"
    #print np.asarray(outputs)

    #IFFT for each M-sized vector
    tx = []
    for out in outputs:
        tx.append(np.fft.ifft(out))
    tx = np.asarray(tx)
    #print "\nOutput of each IFFT round, M-sized vector"
    #print tx


    ####################################
    #              Channel             #
    ####################################

    #Add noise to each OFDM block separately, and calculate the IFFT of the noise (?)
    rx = []
    for burst in tx:
        noise = np.random.normal(scale=scale, size=len(burst.real)) \
            + 1j*np.random.normal(scale=scale, size=len(burst.imag))
        noise = np.fft.ifft(noise)
        rx.append(burst + noise)

    #Parallel to serial
    rx = np.reshape(rx,M*(block/2))

    #noise = np.random.normal(scale=scale, size=len(serialtx.real)) \
    #        + 1j*np.random.normal(scale=scale, size=len(serialtx.imag))
    #noise = np.fft.ifft(noise)
    #rx = serialtx + noise
    #print "\nReceived signal with AWGN noise"
    #print rx

    ####################################
    #             Reception            #
    ####################################

    #Serial to parallel
    rxp = np.reshape(rx,(block/2,M))
    #print "\nSerial to parallel of the received signal"
    #print rxp

    #FFT for each M-sized vector
    y = []
    for received in rxp:
        y.append(np.fft.fft(received))
    y = np.asarray(y)
    #print "\nThe output of the FFT block"
    #print y

    #Demodulation
    demod = []
    for vec in y.T:
        demod.append(fn.qpsk_demodulator(vec))
    demod = np.asarray(demod)
    #print "\nDemodulated streams for each channel"
    #print demod

    #Parallel to Serial
    #print "\nParallel to serial to obtain the original bitstream"
    output = np.reshape(demod.T,M*block)
    #print output
    err = np.where(source != output)
    BER = len(err[0])/(float(M)*block)
    if(plots):
        fn.constellation(y,SNR,BER)

    return BER


####################################
#             Simulation           #
####################################
#ofdmqpsksimulation(-10,True)
#ofdmqpsksimulation(0,True)
#ofdmqpsksimulation(10,True)
simulationBER = []
theoreticalBER = []
for i in range(-20,20,1):
    print "Starting simulation for SNR=" + str(i) + "dB"
    simulationBER.append(ofdmqpsksimulation(i,False))
    theoreticalBER.append(fn.berqpsk(10 ** (i / 10.)))

print simulationBER
#Plots both curves
plt.plot(np.arange(-20,20,1),simulationBER,color="r",marker="o")
#plt.plot(np.arange(-20,20,1),theoreticalBER,color="g",marker="v")
plt.semilogy()

# Creates 2 Rectangles for the legend
p1 = plt.Rectangle((0, 0), 0.1, 0.1, fc="r")
p2 = plt.Rectangle((0, 0), 0.1, 0.1, fc="g")

# Adds the legend into plot
plt.legend((p1, p2), ('Simulation', 'Theoretical'), loc='best')
plt.xlabel("SNR in dB")
plt.ylabel("BER")
plt.grid()
plt.show()