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
    plt.plot(n, h_n)
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
    plt.stem(n, np.imag(h_k))
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
        a_n[ni] = temp
    a_n = a_n * (1 / N)
    # Plot a_n
    plt.figure(figsize=(10, 6))
    plt.plot(n, a_n)
    plt.title('Time-Domain Signal $a(n)$ for Gibbs')
    plt.xlabel('$n$')
    plt.ylabel('$a(n)$')
    plt.grid(True)
    plt.show()


def Exercise12(D=3):
    w = np.linspace(-np.pi, np.pi, 2001)  # We sampled it 2*pi/N times
    N = 2001  # Original number of samples
    n = np.arange(-1000, 1001)

    # Signal a(n)
    a_n = np.where(np.abs(n) < 100, 1, 0)

    # DTFT
    X_1 = np.zeros_like(w, dtype=complex)
    for k in range(N):
        X_1 += a_n[k] * np.exp(-1j * w * k)
    # for k in range(N):
    #     k_shifted = (k + N // 2) % N  # Shift index to center the zero frequency component
    #     X_1[k_shifted] = np.sum(a_n * np.exp(-1j * 2 * np.pi * k * np.arange(N) / N))

    # Plot X(jw)
    plt.figure(figsize=(10, 6))
    plt.plot(w, np.real(X_1))
    plt.title('Transform of $a(n)$')
    plt.xlabel('$k$')
    plt.ylabel('$X(k)$')
    plt.grid(True)
    plt.show()

    # Decimation by a factor of D
    a_dec_2 = a_n[::D]  # Proper decimation, taking every D-th sample
    N_dec = len(a_dec_2)  # New length after decimation

    # New frequency vector for decimated signal
    w_dec = np.linspace(-np.pi, np.pi, N_dec)

    # DTFT of the decimated signal
    X_2 = np.zeros(N_dec, dtype=complex)
    for k2 in range(N_dec):
        X_2 += a_dec_2[k2] * np.exp(-1j * w_dec * k2)
    # for k2 in range(N_dec):
    #     k2_shifted = (k2 + N_dec // 2) % N_dec  # Shift index to center the zero frequency component
    #     X_2[k2_shifted] = np.sum(a_dec_2 * np.exp(-1j * 2 * np.pi * k2 * np.arange(N_dec) / N_dec))

    # Plot DTFT of the decimated signal
    plt.figure(figsize=(10, 6))
    plt.plot(w_dec, np.real(X_2))
    plt.title('Transform of $a_{dec}(n)$ after Decimation by $D$')
    plt.xlabel('$k$')
    plt.ylabel('$X_{dec}(k)$')
    plt.grid(True)
    plt.show()


def Exercise13(padding=3, flag=True):
    N = 2001  # Number of samples
    n = np.arange(-1000, 1001)

    # Signal a(n)
    a_n = np.where(np.abs(n) < 100, 1, 0)
    a_n_padded = []
    for coeff in range(N):
        a_n_padded.append(a_n[coeff])  # Add the a_n coefficient
        if coeff != N - 1:
            a_n_padded.extend([0] * padding)  # Add the padding zeros

    a_n_padded = np.array(a_n_padded)
    N2 = len(a_n_padded)
    w = np.linspace(-np.pi, np.pi, N2)
    X_pad = np.zeros(N2, dtype=complex)
    # DTFT
    for k in range(N2):
        X_pad += a_n_padded[k] * np.exp(-1j * w * k)

    if flag:
        # Plot X_2(k)
        plt.figure(figsize=(10, 6))
        plt.plot(w, np.real(X_pad))
        plt.title('Transform of $X_M(K)$ with Zero Padding')
        plt.xlabel('Frequency: $K$')
        plt.ylabel('$X_M(K)$')
        plt.grid(True)
        plt.show()
    return N2, X_pad, w, a_n_padded


def Exercise14(padding=4, flag=True):
    # We receive the interpolated samples in frequency from exe13
    N2, X_2, w_2, a_n_padded = Exercise13(padding, False)

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
        plt.plot(w_2, H_2)
        plt.title('Frequency Response of the Filter $H_2$')
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    # Inverse DTFT
    N = len(X_2_filtered)
    x_n = np.zeros(N, dtype=complex)
    for n in range(N):
        x_n[n] = np.sum(X_2_filtered * np.exp(1j * w_2 * n)) / N

    if flag:
        # Plot h_n[n]
        plt.figure(figsize=(10, 6))
        plt.plot(n_values, np.real(x_n))
        plt.title('Inverse Transform of $x_M(n)$ with Zero Padding')
        plt.xlabel('$n$')
        plt.ylabel('$x_M(n)$')
        plt.grid(True)
        plt.show()

    return h_n, x_n


def Exercise15(padding=4):
    # Get the filter in time
    h_n, x_n = Exercise14(padding, flag=False)

    # Get the padded signals a_M[n]
    N2, X_pad, w, a_n_padded = Exercise13(padding, flag=False)

    # Now we do a convolution a_n_padded * h_n
    # Lengths of the signals
    len_a = len(a_n_padded)
    len_h = len(h_n)
    len_y = len_a + len_h - 1  # Length of the convolution result

    # Initialize the output array
    y_n = np.zeros(len_y, dtype=complex)

    # Perform the convolution manually
    for n in range(len_y):
        sum_val = 0
        for k in range(len_h):
            if 0 <= n - k < len_a:
                sum_val += a_n_padded[n - k] * h_n[k]
        y_n[n] = sum_val

    plt.figure(figsize=(10, 6))
    plt.stem(w, np.real(h_n))
    plt.title('Filter $h_n$')
    plt.xlabel('$n$')
    plt.ylabel('$h_n$')
    plt.grid(True)
    plt.show()

    # Plot the result of the convolution
    n = np.arange(-(len_y - 1) // 2, (len_y - 1) // 2 + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(n, np.real(y_n))
    plt.title('Convolution Result $a_M(n)$')
    plt.xlabel('$n$')
    plt.ylabel('$a_M(n)$')
    plt.grid(True)
    plt.show()


def Exercise16(padding=3):
    # Get the padded signals a_M[n] from Exercise13
    pad = padding
    N, X_pad, n, a_n = Exercise13(padding=pad, flag=False)
    T = pad + 1

    # ZOH
    n_zoh = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1)
    a_n_ZOH = []
    for i, i_sample in enumerate(a_n):
        if i % T == 0:
            a_n_ZOH.append(i_sample)
        else:
            a_n_ZOH.append(a_n_ZOH[i - 1])

    # Plot the result of the ZOH interpolation
    plt.figure(figsize=(10, 6))
    plt.stem(n_zoh, a_n_ZOH, label='ZOH Interpolated Signal')
    plt.title('ZOH Interpolated Signal $x_{zoh}[n]$')
    plt.xlabel('$n$')
    plt.ylabel('$x_{zoh}[n]$')
    plt.grid(True)
    plt.legend()
    plt.show()

    # FOH
    # Triangle filter
    h_triangle = []
    n_FOH = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1)
    for x in n_FOH:
        if np.abs(x) > T:
            h_triangle.append(0)
        elif - T <= x <= 0:
            h_triangle.append(1 + x / T)
        else:
            h_triangle.append(1 - x / T)
    h_triangle = np.array(h_triangle)

    # Manual convolution
    conv_result = np.zeros(N)
    pad_signal = np.pad(a_n, (N // 2, N // 2), mode='constant')

    for i in range(N):
        conv_result[i] = np.sum(pad_signal[i:i + N] * h_triangle[::-1])

    # Plot the result of the FOH interpolation
    plt.figure(figsize=(10, 6))
    plt.stem(n_FOH, conv_result, label='FOH Interpolated Signal')
    plt.title('FOH Interpolated Signal $x_{foh}[n]$')
    plt.xlabel('$n$')
    plt.ylabel('$x_{foh}[n]$')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    Exercise1()
    #Exercise2()
    #Exercise3()
    #Exercise4()
    #Exercise5()
    #Exercise7()
    #Exercise8()
    #Exercise9()
    #Exercise10()
    #Exercise11()
    #Exercise12()
    #Exercise13()
    #Exercise14()
    #Exercise15()
    #Exercise16()
