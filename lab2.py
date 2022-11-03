from math import sin, pi, sqrt
import matplotlib.pyplot as plt
import numpy as np

# Значения варианта 5
N = 512
K = 3 * N // 4
ph = pi / 8


def value1(x):
    return round(sin(2 * pi * x / N), 3)


def value2(x):
    return round(sin(2 * pi * x / N + ph), 3)


# вычислить среднее квадратическое
def findXA(M, method):
    sum = 0
    for i in range(M):
        sum += (method(i) ** 2)
    return sqrt(1 / (M + 1) * sum)


# вычислить среднее квадратическое
def findXB(M, method):
    sumA = 0
    sumB = 0
    for i in range(M):
        z = method(i)
        sumA += (z ** 2)
        sumB += z
    return sqrt(1 / (M + 1) * sumA - (1 / (M + 1) * sumB) ** 2)


# вычислить амплитуду сигнала
def findAmplitude(M, method):
    y = [method(i) for i in range(M)]
    fft = np.fft.fft(y)
    amplitudes = 2 / M * np.abs(fft)
    return max(amplitudes)


# ======================================================================================================================

def task(method):
    X = []
    Y = [[], [], []]
    for M in range(K, 2 * N):
        deltaXa = findXA(M, method)  # вычислить среднее квадратическое
        deltaXb = findXB(M, method)  # вычислить среднее квадратическое

        amplitude = findAmplitude(M, method)  # вычислить амплитуду сигнала

        deltaA = 0.707 - deltaXa
        deltaB = 0.707 - deltaXb

        delA = 1 - amplitude

        X.append(M)
        Y[0].append(deltaA)
        Y[1].append(deltaB)
        Y[2].append(delA)

    printGraph(X, Y)


def printGraph(X, Y):
    plt.plot(X, Y[0], label='delta by 1')
    plt.plot(X, Y[1], label='delta by 2')
    plt.plot(X, Y[2], label='delta A')
    plt.legend()
    plt.xlabel('M')
    plt.ylabel('Погрешность')
    plt.show()


def main():
    print("start")
    print("======================loading task1...======================")
    task(value1)
    print("======================loading task2...======================")
    task(value2)


if __name__ == '__main__':
    main()
