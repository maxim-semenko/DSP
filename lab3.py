from itertools import repeat
from math import cos, pi, sin, hypot, atan2
from random import randrange

import numpy as np
from matplotlib import pyplot as plt

# Значения из варианта 5
N = 1024
AMPLITUDES = [3, 5, 6, 8, 10, 13, 16]
PHASES = [pi / 6, pi / 4, pi / 3, pi / 2, 3 * pi / 4, pi]
POLY_COUNT = 30

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'darkorange', 'sienna', 'navy', 'blueviolet', 'gray']

labels2_3_4 = ["Оригинальный сигнал", "Восстановленый сигнал", "Амплитудный спектр", "Фазовый спектр"]

labels5_1 = [
    "ВЧ-фильтр",
    "ВЧ-спектр амплитудный",
    "ВЧ-спектр фазовый"]

labels5_2 = ["НЧ-фильтр",
             "НЧ-спектр амплитудный",
             "НЧ-спектр фазовый",
             "Полосовой фильтр",
             "Полосовой фильтр - спектр амплитудный",
             "Полосовой фильтр - спектр фазовый"]


# Тестовый сигнал варианта 5
def signal_point(x, N):
    return 30 * cos(((2 * pi * x) / N) - ((3 * pi) / 4))


# =======================================================================================================================

# для 3.16 и 3.17
def fourier(func, sequence, j, N):
    result = sum(x * func(2 * pi * i * j / N) for i, x in enumerate(sequence))
    return (2 / N) * result


def fourierSpectrum(sequence):
    N = len(sequence)
    spectrum_list = []
    for j in range(N):
        cosine = fourier(cos, sequence, j, N)  # 3.16
        sine = fourier(sin, sequence, j, N)  # 3.17
        amplitude = hypot(sine, cosine)  # 3.18 SQRT (х * х + у * у)
        phase = atan2(sine, cosine)  # 3.19
        spectrum_list.append((amplitude, phase if abs(amplitude) > 0.001 else 0))

    return spectrum_list


def fastFourierSpectrum(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if N == 1:
        return x

    X_even = fastFourierSpectrum(x[::2])
    X_odd = fastFourierSpectrum(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    result = np.concatenate([X_even + factor[:N // 2] * X_odd, X_even + factor[N // 2:] * X_odd])
    return result


def resultFastFourierSpectrum(spectrum):
    fft_result = fastFourierSpectrum(spectrum)
    N = len(spectrum)
    amplitudes = [abs(x) * 2 / N for x in fft_result]
    phases = [np.arctan2(np.imag(x), np.real(x)) if amplitudes[i] > 0.001 else 0 for i, x in enumerate(fft_result)]
    return list(zip(amplitudes, phases))


# 3 б) Восстановить исходный сигнал по спектру
def spectrumPoint(index, j, N, amplitude, phase):
    return amplitude * cos((2 * pi * index * j / N) - phase)


def polyharmonic(index, N, repeats, spectrum):
    return sum(spectrumPoint(index, j, N, spectrum[j][0], spectrum[j][1]) for j in range(repeats))


def restore(spectrum):
    sequence = []
    N = len(spectrum)
    for i in range(N):
        sequence.append(polyharmonic(i, N, N // 2 - 1, spectrum))

    return sequence


def randomsValues(values, length=None):
    range_values = range(length) if length is not None else repeat(0)
    values_len = len(values)
    for _ in range_values:
        yield values[randrange(0, values_len)]


def filter(spectrum, filter_predicate):
    length = len(spectrum)
    halfLength = length // 2

    values = []
    for item in enumerate(spectrum):
        index, value = item
        if index > halfLength:
            index = length - index

        values.append(value if filter_predicate(index) else (0, 0))

    return list(values)


# =======================================================================================================================

def task2():
    original_signal = [signal_point(i, N) for i in range(N)]  # сформировать тестовые сигналы

    spectrum = fourierSpectrum(original_signal)
    amplitude, phase = zip(*spectrum)
    restored_signal = restore(spectrum)  # 2 б) Восстановить исходный сигнал по спектру:
    displayGraph([original_signal, restored_signal, amplitude, phase], labels2_3_4, 2, 2)  # Figure 1 show


def task3(polyharmonic_original_signal):
    spectrum = fourierSpectrum(polyharmonic_original_signal)
    amplitude, phase = zip(*spectrum)  # вычислить амплитудный и фазовый спектр
    restored_signal = restore(spectrum)  # б)Восстановить исходный сигнал по спектру

    displayGraph([polyharmonic_original_signal, restored_signal, amplitude, phase], labels2_3_4, 2, 2)  # Figure 2 show


def task4(polyharmonicOriginalSignal):
    spectrum = resultFastFourierSpectrum(polyharmonicOriginalSignal)  # реализации быстрого преобразования Фурье
    amplitude, phase = zip(*spectrum)
    restored_signal = restore(spectrum)

    displayGraph([polyharmonicOriginalSignal, restored_signal, amplitude, phase], labels2_3_4, 2, 2)  # Figure 3 show


def task5(polyharmonicOriginalSignal):
    spectrum = fourierSpectrum(polyharmonicOriginalSignal)

    spectrumHigh = filter(spectrum, lambda x: x > 10)
    spectrumLow = filter(spectrum, lambda x: x < 10)
    spectrumMedium = filter(spectrum, lambda x: (x > 5) and (x < 10))

    signalHigh = restore(spectrumHigh)
    signalLow = restore(spectrumLow)
    signalMedium = restore(spectrumMedium)

    amplitudeHigh, phaseHigh = zip(*spectrumHigh)
    amplitudeLow, phaseLow = zip(*spectrumLow)
    amplitudeMedium, phaseMedium = zip(*spectrumMedium)

    # Figure 4 show
    displayGraph([signalHigh, amplitudeHigh, phaseHigh], labels5_1, 3, 2)

    # Figure 5 show
    displayGraph([signalLow, amplitudeLow, phaseLow, signalMedium, amplitudeMedium, phaseMedium], labels5_2, 3, 2)


# ==================================ОТРИСОВКА============================================================================
def displayGraph(signals, labels, w, h):
    graph = []
    plt.figure(figsize=(15, 7))
    plt.grid(True)

    for i, signal in enumerate(signals):
        plt.subplot(h, w, i + 1)
        tempGraph, = plt.plot(signal, color=colors[i])
        graph.append(tempGraph)

    plt.figlegend(graph, labels, loc='upper left')
    return graph


def main():
    # Задание 2=========================================================================================================
    task2()

    # Предварительный полигармонический сигнал
    test_spectrum = [(0, 0)] + list(
        zip(list(randomsValues(AMPLITUDES, POLY_COUNT)), list(randomsValues(PHASES, POLY_COUNT))))
    polyharmonic_original_signal = [polyharmonic(i, N, len(test_spectrum), test_spectrum) for i in range(N)]

    # Задание 3=========================================================================================================
    task3(polyharmonic_original_signal)

    # Задание 4=========================================================================================================
    task4(polyharmonic_original_signal)

    # Задание 5=========================================================================================================
    task5(polyharmonic_original_signal)

    plt.show()


if __name__ == '__main__':
    main()
