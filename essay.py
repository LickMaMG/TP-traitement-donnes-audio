import numpy as np
import matplotlib.pyplot as plt
import scipy
import statsmodels.api as sm
import IPython.display

fe, data = scipy.io.wavfile.read('voiceP.wav')
length = data.shape[0] / fe
time = np.linspace(0., length, data.shape[0])

T = 0.04  # durée de stationnarité des signale sonores
N = round(T*fe)
t = np.array(range(-N+1, N-1))/fe

decalage = 0.01
ND = round(decalage*fe)


def calcul_freq_fft(indice_tram, plot_figure):

    x = data[1+int(indice_tram)*ND:N+int(indice_tram)*ND]
    window = scipy.signal.windows.barthann(51, sym=True)

    # x = scipy.signal.convolve(x,window,mode='same')
    X = scipy.fft.fft(x)

    XP = scipy.fft.fftshift(np.abs(X))

    peaks, _ = scipy.signal.find_peaks(XP)
    sorted_peaks = sorted(XP[peaks])[::-1]

    # print(sorted_peaks)
    freq = np.linspace(-fe/2, fe/2, len(XP))

    if plot_figure:
        plt.title("Transformée de Fourier")
        plt.plot(freq, XP)
        plt.show()

    f0 = max(freq[np.argwhere(XP == max(XP))])[0]

    return f0


calcul_freq_fft(0, True)
