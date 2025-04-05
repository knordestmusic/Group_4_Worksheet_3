import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.signal import spectrogram, stft
import scipy.io
from scipy.io import wavfile
from scipy.io.wavfile import write
from scipy import signal
import sounddevice as sd


def Section_1():
    "(1) Sinusoidal Modeling"

    "(1.1) Audio File"

    wavPath = "clarinet_D4_phrase_mezzo-forte_tongued-slur.wav"
    fs, x = wavfile.read(wavPath)
    num_channels = 1 if len(x.shape) == 1 else x.shape[1]
    duration_sample = x.shape[0] / fs

    print(f"samplerate of the audio file is {fs} Hz")
    print(f"number of channels = {num_channels}")
    print(f"duration of the sample is {duration_sample} seconds")

    # OUTPUT
    # samplerate of the audio file is 44100 Hz
    # number of channels = 1
    # duration of the sample is 8.739024943310657 seconds

    "(1.2) Partial Tracking"
    # Note to self - The result of autocorrelation of a signal with itself is always symmetric around lag = 0 (i.e. when the signal is exactly coincident over itself).
    # For example the autocorr of the array [123] is a symmetric array [3 8 14 8 3] where the 3 1st and second members of the resulting array are outcomes of negative lag (lag -2 and -1).
    # Lag of 0 gives 14 (PEAK) and positive lags are 8 and 3. hence if the signal was [1 2 3], the lagmax would be 1 because 8 occurs first in the positive lag phase.

    x = x / max(abs(x))

    nWin = 2048 * 2  # window length
    nHop = 512
    N = int((len(x) - nWin) / nHop)

    frame_matrix = np.zeros([N, nWin])

    for n in range(N):
        frame_n = x[
            n * nHop : n * nHop + nWin
        ]  # stores the samples from starting position n*nHop to n*nHop + nWin
        frame_matrix[n, :] = frame_n

    fmin = 50
    fmax = 500

    minLag = int(fs / fmax)
    maxLag = int(fs / fmin)

    f0_array = np.zeros(N)

    for n in range(N):
        frame = frame_matrix[n, :]
        corr = np.correlate(
            frame, frame, mode="full"
        )  # creates an array of correlation function with values at each lags (lags are basically indices from [ (-N/2)+1 to (N/2)-1 ]
        corr = corr[len(corr) // 2 :]

        peakIndex = minLag + np.argmax(corr[minLag:maxLag])
        # Note to self -> corr(minLag:maxLag) gives us the correlation matrix between the min and max lag values which are relevant to us. argmax gives us the position in
        # the corr array which is the highest peak but it is a relative number. hence we have to add the minLag value to find the exact position at which the peak exists.

        pitch = fs / peakIndex

        f0_array[n] = pitch  # fill the nth indice of the f0 array with the pitch

    plt.plot(f0_array)
    plt.ylim([0, fmax])
    plt.xlabel("Frame number")
    plt.ylabel("Frequency (Hz)")
    plt.title("Fundamental Frequency Trajectory")
    plt.grid(True)
    plt.show()

    nPart = 10

    win = sp.signal.windows.hann(nWin)

    A = np.zeros([N, nPart])
    F = np.zeros([N, nPart])

    nFFT = nWin * 2

    for n in range(N):

        frame_s = frame_matrix[n, :] * win  # multiplying each frame by window

        frame_fft = np.fft.fft(frame_s, n=nFFT)  # perform fft on each windowed frame

        Fabs = abs(frame_fft[0 : int(nFFT / 2)])  # take positive half

        f0 = f0_array[n]  # get the fundamental of this particular frame

        maxRange = int(
            f0 / (fs / nFFT) / 4
        )  # fs / nFFT - gives us frequency resolution per bin, and f0 /(fs/nFFT) is number of bins that represent the fundamental frequency f0. And if we divide it by 4, we search only a quarter of that width

        for i in range(nPart):

            partInd = int((i + 1) * f0 / (fs / nFFT)) # (i+1) because when i=0, we get f0 and i=1 we get first partial i.e. 2f0 and so on...

            if (partInd - maxRange > 0) & (partInd + maxRange <= len(Fabs)):

                a = np.max(Fabs[partInd - maxRange : partInd + maxRange + 1])
                f = (
                    partInd
                    + np.argmax(Fabs[partInd - maxRange : partInd + maxRange + 1])
                    - maxRange
                ) * (fs / nFFT) # np.argmax finds the exact bin within the search window that has the maximum amplitude.

                A[n, i] = a
                F[n, i] = f

    plt.figure(figsize=(10, 5))
    plt.plot(F)
    plt.ylim([0, 5000])
    plt.xlabel('Frame Number')
    plt.ylabel('Frequency (Hz)')
    plt.title('Partial Frequencies Over Time')
    plt.legend([f'Partial {i+1}' for i in range(F.shape[1])], loc='upper right')
    plt.grid(True)
    plt.show()

    # Plot amplitudes
    plt.figure(figsize=(10, 5))
    plt.plot(A)
    plt.xlabel('Frame Number')
    plt.ylabel('Amplitude')
    plt.title('Partial Amplitudes Over Time')
    plt.legend([f'Partial {i+1}' for i in range(A.shape[1])], loc='upper right')
    plt.grid(True)
    plt.show()
    
    np.savetxt('partial_frequencies.sms', F)
    np.savetxt('partial_amplitudes.sms', A)

    # write analysis parameters
    np.savetxt('params.sms', [fs, nHop])


Section_1()  # FOR CALLING THE RESULTS OF SECTION 1
