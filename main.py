import numpy as np
import matplotlib.pyplot as plt


def Exercise1(flag=True):
    # Time vector, we do until 1001 because it's an open interval
    n = np.arange(-1000, 1001)

    # Signal a(n)
    a_n = np.where(np.abs(n) < 100, 1, 0)

    if flag:
        # Plot
        plt.figure(figsize=(10, 6))
        plt.stem(n, a_n, basefmt=" ")
        plt.title('Signal $a(n)$')
        plt.xlabel('$n$')
        plt.ylabel('$a(n)$')
        plt.grid(True)
        plt.show()


def Exercise2(flag=True):
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

    if flag:
        # Plot the magnitude of the Fourier series coefficients
        plt.figure(figsize=(10, 6))
        plt.stem(k, a_k, basefmt=" ")
        plt.title('Fourier Series Coefficients $a_k$')
        plt.xlabel('$k$')
        plt.ylabel('$a_k$')
        plt.ylim(-0.12, 0.12)
        plt.grid(True)
        plt.show()
    return a_k, N, n


def Exercise3(flag=True):
    # Use the results from Exercise 2
    a_k, N, n = Exercise2(False)

    # Time shift
    k = np.arange(-1000, 1001)
    shift = 100
    b_k = a_k * np.exp(-1j * 2 * np.pi * k * shift / N)

    # Compute the inverse Fourier transform to obtain the time-shifted signal
    b_n = np.zeros_like(n, dtype=complex)
    for ni in range(len(n)):
        b_n[ni] = np.sum(b_k * np.exp(1j * 2 * np.pi * k * n[ni] / N))

    if flag:
        # Plot the time-shifted signal
        plt.figure(figsize=(10, 6))
        plt.stem(n, b_n, basefmt=" ")
        plt.title('Time-Shifted Signal $b(n)$')
        plt.xlabel('$n$')
        plt.ylabel('$b(n)$')
        plt.grid(True)
        plt.show()

    return n, b_n, b_k


def Exercise4():
    # Use the results from Exercise 2
    a_k, N, n = Exercise2(False)

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
    a_k, N, n = Exercise2(False)

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


def Exercise7():
    # Time vector, we do until 1001 because it's an open interval
    N = 2001
    n1 = np.arange(-1000, 1001)

    # Signal a(n)
    a_n = np.where(np.abs(n1) < 100, 1, 0)

    # Signal b(n)
    n2, b_n, b_k = Exercise3(False)

    # Signal e(n)
    e_n = a_n * b_n

    # Transfer e_n -> e_k
    k = np.arange(-1000, 1001)
    e_k = np.zeros(N, dtype=complex)
    for ki in range(-1000, 1001):
        for x in range(-1000, 1001):
            e_k[ki + 1000] += e_n[x + 1000] * np.exp(-1j * ki * 2 * np.pi * x / N)
        e_k[ki + 1000] *= (1 / N)

    # Transfer a_n -> a_k
    a_k, N, n = Exercise2(False)

    # Cyclic convolution a_(l)**b_(k-l)
    e_kk = np.zeros(N, dtype=complex)
    for ki in range(-1000, 1001):
        for x in range(-1000, 1001):
            e_kk[ki + 1000] += a_k[x] * b_k[ki - x]

    # Plot e_k and e_kk side by side
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Plot e_k
    axs[0].stem(k, e_k, basefmt=" ")
    axs[0].set_title('Fourier Series Coefficients $e_k$ - Multiplication in Time')
    axs[0].set_xlabel('$k$')
    axs[0].set_ylabel('$e_k$')
    axs[0].grid(True)

    # Plot e_kk
    axs[1].stem(k, e_kk, basefmt=" ")
    axs[1].set_title('Fourier Series Coefficients $e_(kk)$ - Cyclic Convolution')
    axs[1].set_xlabel('$k$')
    axs[1].set_ylabel('$e_(kk)$')
    axs[1].grid(True)

    # Set common y-axis limits if desired
    max_ylim = max(np.max(np.abs(e_k)), np.max(np.abs(e_kk))) * 1.1
    axs[0].set_ylim(-max_ylim, max_ylim)
    axs[1].set_ylim(-max_ylim, max_ylim)

    plt.tight_layout()
    plt.show()


def Exercise8(flag=True):
    # Time vector, we do until 1001 because it's an open interval
    n = np.arange(-1000, 1001)
    N = 2001

    # Signal a(n)
    a_n = np.where(np.abs(n) < 100, 1, 0)

    # Signal g(n)
    g_n = a_n * np.cos(2 * np.pi * 500 * n / N)

    # Transfer g_n -> g_k
    k = np.arange(-1000, 1001)
    g_k = np.zeros(N, dtype=complex)
    for ki in range(-1000, 1001):
        for x in range(-1000, 1001):
            g_k[ki + 1000] += g_n[x + 1000] * np.exp(-1j * ki * 2 * np.pi * x / N)
        g_k[ki + 1000] *= (1 / N)

    if flag:
        # Plot the convoluted signal
        plt.figure(figsize=(10, 6))
        plt.stem(n, np.real(g_k), basefmt=" ")
        plt.title('Convoluted Signal $g(k)$')
        plt.xlabel('$k$')
        plt.ylabel('$g(k)$')
        plt.grid(True)
        plt.show()
    return g_k


def Exercise9():
    # Time vector, we do until 1001 because it's an open interval
    n = np.arange(-1000, 1001)
    N = 2001

    # Signal a(n)
    a_n = np.where(np.abs(n) < 100, 1, 0)

    # Signal h(n)
    h_n = a_n * np.sin(2 * np.pi * 500 * n / N)

    # Plot the original h_n signal
    plt.figure(figsize=(10, 6))
    plt.stem(n, h_n, basefmt=" ")
    plt.title('Original Signal $h(n)$')
    plt.xlabel('$n$')
    plt.ylabel('$h(n)$')
    plt.grid(True)
    plt.show()

    # Transfer h_n -> h_k
    k = np.arange(-1000, 1001)
    h_k = np.zeros(N, dtype=complex)
    for ki in range(-1000, 1001):
        for x in range(-1000, 1001):
            h_k[ki + 1000] += h_n[x + 1000] * np.exp(-1j * ki * 2 * np.pi * x / N)
        h_k[ki + 1000] *= (1 / N)

    # Plot the transformed h_k signal
    plt.figure(figsize=(10, 6))
    plt.stem(n, h_k, basefmt=" ")
    plt.title('Transformed Signal $h(k)$')
    plt.xlabel('$k$')
    plt.ylabel('$h(k)$')
    plt.grid(True)
    plt.show()

    # Let's bring g_k from exe8
    g_k = Exercise8(False)

    # calculate H_k
    H_k = np.zeros_like(k, dtype=complex)
    H_k = -1j * np.sign(k)
    H_k[1000] = 0

    # calculate h_kk = H_k * g_k
    h_kk = H_k * g_k

    # Plot the Hilbert signal
    plt.figure(figsize=(10, 6))
    plt.stem(k, np.imag(h_kk), basefmt=" ")
    plt.title('After Hilbert Signal $h(k)$')
    plt.xlabel('$k$')
    plt.ylabel('$h(k)$')
    plt.grid(True)
    plt.show()

    # now calculate h_nn which is inverse transform of h_kk
    h_nn = np.zeros_like(n, dtype=complex)
    for ni in range(len(n)):
        h_nn[ni] = np.sum(h_kk * np.exp(1j * 2 * np.pi * k * n[ni] / N))

    # Plot the Transformed back Hilbert signal
    plt.figure(figsize=(10, 6))
    plt.stem(n, h_nn, basefmt=" ")
    plt.title('After Hilbert Signal in time $h(n)$')
    plt.xlabel('$n$')
    plt.ylabel('$h(n)$')
    plt.grid(True)
    plt.show()


def Exercise10():
    # Retrieve a_k from Exercise2
    a_k, N, n = Exercise2(False)

    # Define padding and compute the padded Fourier coefficients f_k
    padding = 4
    f_k = []
    for coeff in a_k:
        f_k.append(coeff)  # Add the a_k coefficient
        f_k.extend([0] * padding)  # Add the padding zeros

    f_k = np.array(f_k)  # Convert to numpy array
    f_k = 0.2 * f_k

    # Determine the new length after padding
    L = len(f_k)

    # Define a new time vector matching the length of f_k
    n_f = np.arange(-5000, 5005)

    # Compute the inverse Fourier transform to obtain the time-shifted signal
    f_n = np.zeros_like(n, dtype=complex)
    for ni in range(len(n)):
        f_n[ni] = np.sum(f_k * np.exp(1j * 2 * np.pi * n_f * n[ni] / N))

    # Plot the real part of the time-domain signal f_n
    plt.figure(figsize=(10, 6))
    plt.stem(n, f_n, basefmt=" ")
    plt.title('Time-Domain Signal $f(n)$ After Zero Padding')
    plt.xlabel('$n$')
    plt.ylabel('$f(n)$')
    plt.grid(True)
    plt.show()


def Exercise11():
    # Define M (range of summation)
    M = 950

    # Retrieve a_k from Exercise2
    a_k, N, n = Exercise2(False)

    # Define the range for k values
    k = np.arange(-M, M + 1)
    a_n = np.zeros_like(n, dtype=complex)

    # Compute the partial inverse Fourier series sum
    for ni in range(len(n)):
        temp = 0
        for m in k:
            temp += a_k[m + 1000] * np.exp(1j * 2 * np.pi * m * n[ni] / N)
        a_n[ni] = (1 / N) * temp

    # Plot a_n
    plt.figure(figsize=(10, 6))
    plt.stem(n, a_n, basefmt=" ")
    plt.title('Time-Domain Signal $a(n)$ for Gibbs')
    plt.xlabel('$n$')
    plt.ylabel('$a(n)$')
    plt.grid(True)
    plt.show()


def Exercise12(D=3):
    w = np.linspace(-np.pi, np.pi, 2001)  # We sampled it 2*pi/N times
    N = 2001  # Number of samples
    n = np.arange(-1000, 1001)

    # Signal a(n)
    a_n = np.where(np.abs(n) < 100, 1, 0)

    # It's the formula for DTFT, but we actually do a DFT
    # when we sample on a single period of DTFT between -pi to pi
    X_1 = np.zeros_like(w, dtype=complex)
    for k in range(N):
        X_1 += a_n[k] * np.exp(-1j * w * k)

    # Plot X(jw)
    plt.figure(figsize=(10, 6))
    plt.stem(w, X_1, basefmt=" ")
    plt.title('DFT of $a(n)$')
    plt.xlabel('$k$')
    plt.ylabel('$X(k)$')
    plt.grid(True)
    plt.show()

    # Decimation by a factor of D that we chose and DFT after it
    a_dec_2 = np.zeros(N)
    a_dec_2[::D] = a_n[::D]  # Sample every D indexes

    X_2 = np.zeros(N, dtype=complex)
    for k2 in range(N):
        X_2 += a_dec_2[k2] * np.exp(-1j * w * k2)

    plt.figure(figsize=(10, 6))
    plt.stem(w, abs(X_2), basefmt=" ")
    plt.title('DFT of $a_2(n)$')
    plt.xlabel('$k$')
    plt.ylabel('$X(k)$')
    plt.grid(True)
    plt.show()


def Exercise13(padding=2, flag=True):
    N = 2001  # Number of samples
    n = np.arange(-1000, 1001)

    # Signal a(n)
    a_n = np.where(np.abs(n) < 100, 1, 0)
    a_n_padded = []
    for coeff in range(N):
        a_n_padded.append(a_n[coeff])  # Add the a_k coefficient
        if coeff != N - 1:
            a_n_padded.extend([0] * padding)  # Add the padding zeros

    a_n_padded = np.array(a_n_padded)
    N2 = len(a_n_padded)
    w = np.linspace(-np.pi, np.pi, N2)
    X_pad = np.zeros(N2, dtype=complex)
    for k in range(N2):
        X_pad += a_n_padded[k] * np.exp(-1j * w * k)

    if flag:
        # Plot X_2(k)
        plt.figure(figsize=(10, 6))
        plt.stem(w, X_pad)
        plt.title('DFT of $X_M(K)$ with Zero Padding')
        plt.xlabel('Frequency: $K$')
        plt.ylabel('$X_M(K)$')
        plt.grid(True)
        plt.show()
    return N2, X_pad, w


def Exercise14(padding=2, flag=True):
    # We receive the interpolated samples in frequency from exe13
    N2, X_2, w_2 = Exercise13(padding, False)

    # Now build the filters
    cutoff_freq = np.pi / padding
    # Create the filter in frequency domain
    H_2 = np.where(np.abs(w_2) <= cutoff_freq, 1, 0)
    X_2_filtered = X_2 * H_2

    # Calculate filter in time
    N = len(H_2)
    h_n = np.zeros(N, dtype=complex)
    n_values = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1)
    for n_idx, n in enumerate(n_values):
        h_n[n_idx] = np.sum(H_2 * np.exp(1j * w_2 * n)) / N

    if flag:
        # Plot the filter
        plt.figure(figsize=(10, 6))
        plt.stem(n_values, h_n)
        plt.title('$H_M(K)$')
        plt.xlabel('Frequency: $K$')
        plt.ylabel('$H_M(K)$')
        plt.grid(True)
        plt.show()

    # Inverse DFT
    N = len(X_2_filtered)
    x_n = np.zeros(N, dtype=complex)
    for n in range(N):
        x_n[n] = np.sum(X_2_filtered * np.exp(1j * w_2 * n)) / N

    if flag:
        # Plot X_2(k)
        plt.figure(figsize=(10, 6))
        plt.plot(n_values, x_n)
        plt.title('DFT of $X_M(K)$ with Zero Padding')
        plt.xlabel('Frequency: $K$')
        plt.ylabel('$X_M(K)$')
        plt.grid(True)
        plt.show()

    return H_2, h_n, x_n


def Exercise15():
    # TODO
    pass


def Exercise16():
    # TODO
    pass


def Exercise17():
    # TODO
    pass


if __name__ == '__main__':
    # Exercise1()
    # Exercise2()
    # Exercise3()
    # Exercise4()
    # Exercise5()
    # Exercise7()
    # Exercise8()
    # Exercise9()
    # Exercise10()
    # Exercise11()
    # Exercise12()
    # Exercise13()
    Exercise14()
    # Exercise15()
    # Exercise16()
    # Exercise17()
