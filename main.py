import numpy as np
import matplotlib.pyplot as plt


def Exercise1():
    # Time vector, we do until 1001 because it's an open interval
    n = np.arange(-1000, 1001)

    # Signal a(n)
    a_n = np.where(np.abs(n) < 100, 1, 0)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.stem(n, a_n, basefmt=" ")
    plt.title('Signal $a(n)$')
    plt.xlabel('$n$')
    plt.ylabel('$a(n)$')
    plt.grid(True)
    plt.show()


def Exercise2():
    # Define the time vector
    n = np.arange(-1000, 1001)
    N = 2001  # Period of the signal
    a_n = np.where(np.abs(n) < 100, 1, 0)

    # Compute the Fourier series coefficients
    k = np.arange(-1000, 1001)
    a_k = np.zeros(N, dtype=complex)
    for ki in range(-1000, 1001):
        for x in range(-1000, 1001):
            a_k[ki + 1000] += a_n[x + 1000] * np.exp(-1j * ki * 2 * np.pi * x / N)
        a_k[ki + 1000] *= (1 / N)

    # Plot the magnitude of the Fourier series coefficients
    plt.figure(figsize=(10, 6))
    plt.stem(k, a_k, basefmt=" ")
    plt.title('Magnitude of Fourier Series Coefficients $a_k$')
    plt.xlabel('$k$')
    plt.ylabel('$a_k$')
    plt.ylim(-0.12, 0.12)
    plt.grid(True)
    plt.show()
    return a_k, N, n


def Exercise3():
    # Use the results from Exercise 2
    a_k, N, n = Exercise2()

    # Time shift
    k = np.arange(-1000, 1001)
    shift = 100
    b_k = a_k * np.exp(-1j * 2 * np.pi * k * shift / N)

    # Compute the inverse Fourier transform to obtain the time-shifted signal
    b_n = np.zeros_like(n, dtype=complex)
    for ni in range(len(n)):
        b_n[ni] = np.sum(b_k * np.exp(1j * 2 * np.pi * k * n[ni] / N))

    # Plot the time-shifted signal
    plt.figure(figsize=(10, 6))
    plt.stem(n, b_n, basefmt=" ")
    plt.title('Time-Shifted Signal $b(n)$')
    plt.xlabel('$n$')
    plt.ylabel('$b(n)$')
    plt.grid(True)
    plt.show()


def Exercise4():
    # Use the results from Exercise 2
    a_k, N, n = Exercise2()

    # Multiply by k
    k = np.arange(-1000, 1001)
    c_k = a_k * np.exp(-1j * 2 * np.pi * k / N)
    c_k = a_k - c_k

    # Compute the inverse Fourier transform to obtain the time-shifted signal
    c_n = np.zeros_like(n, dtype=complex)
    for ni in range(len(n)):
        c_n[ni] = np.sum(c_k * np.exp(1j * 2 * np.pi * k * n[ni] / N))

    # Plot the time-shifted signal
    plt.figure(figsize=(10, 6))
    plt.stem(n, c_n, basefmt=" ")
    plt.title('Derivative Signal $c(n)$')
    plt.xlabel('$n$')
    plt.ylabel('$c(n)$')
    plt.grid(True)
    plt.show()


def Exercise5():
    # Use the results from Exercise 2
    a_k, N, n = Exercise2()

    # Multiplication in spectrum (times N)
    k = np.arange(-1000, 1001)
    d_k = a_k * a_k
    d_k = d_k * N

    # Compute the inverse Fourier transform
    d_n = np.zeros_like(n, dtype=complex)
    for ni in range(len(n)):
        d_n[ni] = np.sum(d_k * np.exp(1j * 2 * np.pi * k * n[ni] / N))

    # Plot the convoluted signal
    plt.figure(figsize=(10, 6))
    plt.stem(n, d_n, basefmt=" ")
    plt.title('Convoluted Signal $c(n)$')
    plt.xlabel('$n$')
    plt.ylabel('$d(n)$')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    Exercise1()
    # Exercise2()
    # Exercise3()
    # Exercise4()
    # Exercise5()




