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
SNR=10

def bersimulation(SNR):
    # Initial setup
    # Eb/N0
    EbN0 = 10 ** (SNR / 10.)
    # N0 (The dot is to force the division to be a float
    N0 = 1. / EbN0
    # Noise scale
    scale = np.sqrt(N0 / 2)
    #Initialize the number of errors
    errors = 0
    BERfinal = 0.0
    totalsent = 0
    while(errors<100):
        ####################################
        #           Transmission           #
        ####################################

        # We generate M * block random bits
        source = np.round(np.random.rand(1, int(M * block)))[0]

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
            tx.append(np.fft.ifft(out))
        tx = np.asarray(tx)

        ####################################
        #              Channel             #
        ####################################

        # Add noise to each OFDM block separately, and calculate the IFFT of the noise (?)
        rx = []
        for burst in tx:
            noise = np.random.normal(scale=scale, size=len(burst.real)) \
                    + 1j * np.random.normal(scale=scale, size=len(burst.imag))
            noise = np.fft.ifft(noise)
            rx.append(burst + noise)

        # Parallel to serial
        rx = np.reshape(rx, M * (block / 2))

        ####################################
        #             Reception            #
        ####################################

        # Serial to parallel
        rxp = np.reshape(rx, (block / 2, M))

        # FFT for each M-sized vector
        y = []
        for received in rxp:
            y.append(np.fft.fft(received))
        y = np.asarray(y)

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
        totalsent += M*block
    BER = errors / float(totalsent)
    return BER


####################################
#             Simulation           #
####################################
#ofdmqpsksimulation(-10,True)
#ofdmqpsksimulation(0,True)
#ofdmqpsksimulation(10,True)
simulationBER = []
theoreticalBER = []
range = np.arange(-20,12,1)
for i in range:
    print "Starting simulation for SNR=" + str(i) + "dB"
    BER = bersimulation(i)
    print BER
    simulationBER.append(BER)
    theoreticalBER.append(fn.berqpsk(10 ** (i / 10.)))

#Plots both curves
plt.plot(range,simulationBER,color="r",marker="o")
plt.plot(range,theoreticalBER,color="g",marker="v")
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