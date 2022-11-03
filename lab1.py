import matplotlib.pyplot as plt
import numpy as np

PI = np.pi
N = 1024

# Вариант 5

# Задание 2а
task2a_A = 7
task2a_F = 5
task2a_phases = [PI, 0, PI / 3, PI / 6, PI / 2]

# Задание 2б
task2b_A = 5
task2b_phase = 3 * PI / 4
task2b_F = [1, 5, 11, 6, 3]

# Задание 2c
task2c_F = 3
task2c_phase = 3 * PI / 4
task2c_A = [1, 2, 11, 4, 2]

# Задание 3
task3_A = [9, 9, 9, 9, 9]
task3_F = [1, 2, 3, 4, 5]
task3_phases = [PI / 2, 0, PI / 4, PI / 3, PI / 6]

# Задание 4
task4_A = [5, 5, 5, 5, 5]
task4_F = [10, 20, 30, 40, 50]
task4_phases = [PI / 9, PI / 4, PI / 3, PI / 6, 0]


def harmonic(A, F, amount, phase, graph):
    n = np.arange(0, amount + 1, 1)
    result = (A * np.sin(((2 * PI * F * n) / amount) + phase))
    graph.plot(n, result)


def polyharmonic(A, F, amount, phases, harmonics, graph):
    n = np.arange(0, amount + 1, 1)
    result = 0
    for i in range(harmonics):
        result += (A[i] * np.sin(((2 * PI * F[i] * n) / amount) + phases[i]))

    graph.plot(n, result)


def change(start, x, max):
    return max * start * x + start


def polyharmomic_task4(ampls, freqs, phases, max, amount, harmonics, graph):
    n = np.arange(0, amount + 1, 1)
    result = 0
    for i in range(harmonics):
        result += change(ampls[i], n / amount, max) * np.sin(
            ((2 * PI * change(freqs[i], n / amount, max) * n) / amount)
            + change(phases[i], n / amount, max))

    graph.plot(n, result)


def main():
    fig = plt.figure(figsize=(25, 10), constrained_layout=True, dpi=90)
    spec = fig.add_gridspec(2, 4)

    graph_task2_a = fig.add_subplot(spec[0, 0])
    graph_task2_b = fig.add_subplot(spec[0, 1])
    graph_task2_c = fig.add_subplot(spec[0, 2])
    graph_task3 = fig.add_subplot(spec[1, 0])
    graph_task4 = fig.add_subplot(spec[1, 1])

    # 2a
    for phase in task2a_phases:
        harmonic(task2a_A, task2a_F, N, phase, graph_task2_a)

    # 2b
    for freq in task2b_F:
        harmonic(task2b_A, freq, N, task2b_phase, graph_task2_b)

    # 2c
    for ampl in task2c_A:
        harmonic(ampl, task2c_F, N, task2c_phase, graph_task2_c)

    # 3
    polyharmonic(task3_A, task3_F, N, task3_phases, 5, graph_task3)

    # 4
    polyharmomic_task4(task4_A, task4_F, task4_phases, 0.2, N, 5, graph_task4)

    plt.show()


if __name__ == '__main__':
    main()
