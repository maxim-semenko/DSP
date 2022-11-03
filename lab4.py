from random import choice
from math import sin, pi, hypot, atan2, cos
from statistics import mean
from matplotlib import pyplot as plt

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
labels = ["Оригинальный сигнал", "Скользящее усреднение", "Парабола 4 степени", "Медианная фильтрация"]

# Вариант 5
N = 512
AVERAGED_WINDOW = 3  # Размер окна скользящего усреднения
MEDIAN_WINDOW = 7  # Размер окна медианной фильтрации


# ===================================================ФУРЬЕ==============================================================

def fourier(func, sequence, j, N):
    result = sum(x * func(2 * pi * i * j / N) for i, x in enumerate(sequence))
    return (2 / N) * result


def fourier_spectrum(sequence):
    N = len(sequence)
    spectrum_list = []
    for j in range(N // 2):
        cosine = fourier(cos, sequence, j, N)
        sine = fourier(sin, sequence, j, N)
        amplitude = hypot(sine, cosine)
        phase = atan2(sine, cosine)
        spectrum_list.append((amplitude, phase if abs(amplitude) > 0.001 else 0))

    return spectrum_list


# ===================================================LABA_4=============================================================

# 1. Сформировать сигнал для исследований:
def generateSignals(hi, N, B1, B2):
    return B1 * sin((2 * pi * hi) / N) + sum(B2 * choice([-1, 1]) * sin((2 * pi * hi * j) / N) for j in range(50, 71))


# скользящим усреднением с окном сглаживания
def averagedFilter(values, window):
    m = (window - 1) // 2  # смещение
    range_values = range(len(values))

    return [mean(values[j] for j in range(i - m, i + m) if j in range_values) for i in range_values]


# Метод параболы 4й степени
def parabolaFilter(values):
    range_values = range(len(values))
    get_value = lambda i: values[i] if i in range_values else 0

    def points(i):
        return 1 / 2431 * (110 * get_value(i - 6) - 198 * get_value(i - 5) - 135 * get_value(i - 4)
                           + 110 * get_value(i - 3) + 390 * get_value(i - 2) + 600 * get_value(i - 1)
                           + 677 * get_value(i) + 600 * get_value(i + 1) + 390 * get_value(i + 2)
                           + 110 * get_value(i + 3) - 135 * get_value(i + 4) - 198 * get_value(i + 5)
                           + 110 * get_value(i + 6))

    return [points(i) for i in range_values]


# медианная фильтрация
def medianFilter(values, window):
    m = (window - 1) // 2
    range_values = range(len(values))

    def points(i):
        window_values = [values[j] for j in range(i - m, i + m) if j in range_values]
        window_values = sorted(window_values)
        return mean(window_values) if window_values else 0

    return [points(i) for i in range_values]


# Отрисовка графиков
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
    # 1. Сформировать сигнал для исследований
    random_signal = [generateSignals(i, N, 99, 2) for i in range(N)]

    # скользящим усреднением с окном сглаживания в соответствии с вариантом задания
    moving_averaged = averagedFilter(random_signal, AVERAGED_WINDOW)

    # параболой четвертой степени
    parabola = parabolaFilter(random_signal)

    # медианной фильтрацией с размером окна в соответствии с вариантом задания
    moving_median = medianFilter(random_signal, MEDIAN_WINDOW)

    # 2 и 4 Спектры сигналов и сглаженных сигналов
    amplitude, phase = zip(*fourier_spectrum(random_signal))
    amplitude_averaged, phase_averaged = zip(*fourier_spectrum(moving_averaged))
    amplitude_parabola, phase_parabola = zip(*fourier_spectrum(parabola))
    amplitude_median, phase_median = zip(*fourier_spectrum(moving_median))

    # Вывод графиков сигналов
    displayGraph([random_signal, moving_averaged, parabola, moving_median], labels, 2, 2)

    # Вывод амплитуд сигналов
    displayGraph([amplitude, amplitude_averaged, amplitude_parabola, amplitude_median], labels, 2, 2)

    # Вывод фаз сигналов
    displayGraph([phase, phase_averaged, phase_parabola, phase_median], labels, 2, 2)

    plt.show()


if __name__ == '__main__':
    main()
