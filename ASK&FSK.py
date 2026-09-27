import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

bits_input = input("Enter binary data (e.g. 10110010): ").strip()

if not bits_input or not all(bit in '01' for bit in bits_input):
    raise ValueError("Enter a valid binary sequence containing only 0 and 1.")

bits = np.array([int(bit) for bit in bits_input])

Rb = float(input("Enter bit rate Rb (bits/sec): "))
Fs = float(input("Enter sampling frequency Fs (Hz): "))

fc_ask = float(input("Enter ASK carrier frequency fc (Hz): "))

f0 = float(input("Enter FSK frequency f0 for bit 0 (Hz): "))
f1 = float(input("Enter FSK frequency f1 for bit 1 (Hz): "))

snr_db = float(input("Enter SNR (dB): "))

Nbits_ber = int(
    input("Enter number of bits for BER simulation: ")
)

Tb = 1 / Rb
Ns = int(Fs * Tb)

if Ns < 2:
    raise ValueError(
        "Sampling frequency is too low. Increase Fs."
    )

if fc_ask >= Fs / 2:
    raise ValueError(
        "ASK carrier must be less than Fs/2."
    )

if f0 >= Fs / 2 or f1 >= Fs / 2:
    raise ValueError(
        "FSK frequencies must be less than Fs/2."
    )


delta_f = abs(f1 - f0)

print("\n============================================")
print("SYSTEM PARAMETERS")
print("============================================")

print("Input bits       :", bits_input)
print("Number of bits   :", len(bits))
print("Bit rate Rb      :", Rb, "bits/sec")
print("Bit duration Tb  :", Tb, "sec")
print("Sampling freq.   :", Fs, "Hz")
print("ASK carrier      :", fc_ask, "Hz")
print("FSK f0           :", f0, "Hz")
print("FSK f1           :", f1, "Hz")
print("Tone spacing     :", delta_f, "Hz")
print("SNR              :", snr_db, "dB")

print("\nFSK orthogonality condition:")
print("Delta_f = k / Tb")

print("Minimum orthogonal spacing =",
      1 / Tb, "Hz")

if np.isclose(
    delta_f * Tb,
    round(delta_f * Tb)
):
    print("Selected FSK tones are orthogonal.")
else:
    print("Selected FSK tones are NOT orthogonal.")


# ============================================================
# ASK TRANSMITTER
# ============================================================

def ask_transmitter(bits, fc, Fs, Tb):

    Ns = int(Fs * Tb)

    t_bit = np.arange(Ns) / Fs

    carrier = np.cos(
        2 * np.pi * fc * t_bit
    )

    signal = np.concatenate([
        carrier if bit == 1
        else np.zeros(Ns)
        for bit in bits
    ])

    t = np.arange(len(signal)) / Fs

    return signal, t


# ============================================================
# FSK TRANSMITTER
# ============================================================

def fsk_transmitter(bits, f0, f1, Fs, Tb):

    Ns = int(Fs * Tb)

    t_bit = np.arange(Ns) / Fs

    signal = np.concatenate([
        np.cos(
            2 * np.pi *
            (f1 if bit == 1 else f0) *
            t_bit
        )
        for bit in bits
    ])

    t = np.arange(len(signal)) / Fs

    return signal, t


# ============================================================
# AWGN CHANNEL
# ============================================================

def awgn(signal, snr_db):

    signal_power = np.mean(signal ** 2)

    snr_linear = 10 ** (snr_db / 10)

    noise_power = (
        signal_power / snr_linear
    )

    noise = np.sqrt(
        noise_power
    ) * np.random.randn(len(signal))

    return signal + noise


# ============================================================
# ASK COHERENT CORRELATOR RECEIVER
# ============================================================

def ask_coherent_receiver(
    rx, fc, Fs, Tb
):

    Ns = int(Fs * Tb)

    Nbits = len(rx) // Ns

    t_bit = np.arange(Ns) / Fs

    reference = np.cos(
        2 * np.pi * fc * t_bit
    )

    statistics = []

    for i in range(Nbits):

        r = rx[
            i * Ns:(i + 1) * Ns
        ]

        correlation = np.sum(
            r * reference
        ) / Ns

        statistics.append(correlation)

    statistics = np.array(statistics)

    # Decision threshold
    threshold = 0.25

    detected_bits = (
        statistics > threshold
    ).astype(int)

    return detected_bits, statistics


# ============================================================
# FSK COHERENT CORRELATOR RECEIVER
# ============================================================

def fsk_coherent_receiver(
    rx, f0, f1, Fs, Tb
):

    Ns = int(Fs * Tb)

    Nbits = len(rx) // Ns

    t_bit = np.arange(Ns) / Fs

    reference_0 = np.cos(
        2 * np.pi * f0 * t_bit
    )

    reference_1 = np.cos(
        2 * np.pi * f1 * t_bit
    )

    corr0 = []
    corr1 = []

    detected_bits = []

    for i in range(Nbits):

        r = rx[
            i * Ns:(i + 1) * Ns
        ]

        z0 = np.sum(
            r * reference_0
        ) / Ns

        z1 = np.sum(
            r * reference_1
        ) / Ns

        corr0.append(z0)
        corr1.append(z1)

        if z1 > z0:
            detected_bits.append(1)
        else:
            detected_bits.append(0)

    return (
        np.array(detected_bits),
        np.array(corr0),
        np.array(corr1)
    )

def fsk_noncoherent_receiver(
    rx, f0, f1, Fs, Tb
):

    Ns = int(Fs * Tb)

    Nbits = len(rx) // Ns

    t_bit = np.arange(Ns) / Fs

    # References for f0
    cos0 = np.cos(
        2 * np.pi * f0 * t_bit
    )

    sin0 = np.sin(
        2 * np.pi * f0 * t_bit
    )

    # References for f1
    cos1 = np.cos(
        2 * np.pi * f1 * t_bit
    )

    sin1 = np.sin(
        2 * np.pi * f1 * t_bit
    )

    energy0 = []
    energy1 = []

    detected_bits = []

    for i in range(Nbits):

        r = rx[
            i * Ns:(i + 1) * Ns
        ]

        I0 = np.sum(
            r * cos0
        ) / Ns

        Q0 = np.sum(
            r * sin0
        ) / Ns

        I1 = np.sum(
            r * cos1
        ) / Ns

        Q1 = np.sum(
            r * sin1
        ) / Ns

        E0 = I0 ** 2 + Q0 ** 2
        E1 = I1 ** 2 + Q1 ** 2

        energy0.append(E0)
        energy1.append(E1)

        if E1 > E0:
            detected_bits.append(1)
        else:
            detected_bits.append(0)

    return (
        np.array(detected_bits),
        np.array(energy0),
        np.array(energy1)
    )


# ============================================================
# POWER SPECTRUM
# ============================================================

def power_spectrum(signal, Fs):

    N = len(signal)

    X = np.fft.fft(signal)

    frequency = np.fft.fftfreq(
        N,
        1 / Fs
    )

    positive = frequency >= 0

    frequency = frequency[positive]

    power = (
        np.abs(X[positive]) ** 2
        / N
    )

    return frequency, power

def calculate_ber(original, detected):

    return np.mean(
        original != detected
    )



ask_tx, t_ask = ask_transmitter(
    bits,
    fc_ask,
    Fs,
    Tb
)

fsk_tx, t_fsk = fsk_transmitter(
    bits,
    f0,
    f1,
    Fs,
    Tb
)

ask_rx = awgn(
    ask_tx,
    snr_db
)

fsk_rx = awgn(
    fsk_tx,
    snr_db
)


ask_detected, ask_statistics = \
    ask_coherent_receiver(
        ask_rx,
        fc_ask,
        Fs,
        Tb
    )


(
    fsk_coherent_detected,
    corr0,
    corr1
) = fsk_coherent_receiver(
    fsk_rx,
    f0,
    f1,
    Fs,
    Tb
)


(
    fsk_noncoherent_detected,
    energy0,
    energy1
) = fsk_noncoherent_receiver(
    fsk_rx,
    f0,
    f1,
    Fs,
    Tb
)


print("\n============================================")
print("TRANSMISSION RESULTS")
print("============================================")

print("\nOriginal bits:")
print(bits_input)

print("\nASK coherent detected bits:")
print(
    ''.join(
        map(str, ask_detected)
    )
)

print("\nFSK coherent detected bits:")
print(
    ''.join(
        map(str, fsk_coherent_detected)
    )
)

print("\nFSK noncoherent detected bits:")
print(
    ''.join(
        map(str, fsk_noncoherent_detected)
    )
)

print("\n--------------------------------------------")

print(
    "ASK coherent BER =",
    calculate_ber(
        bits,
        ask_detected
    )
)

print(
    "FSK coherent BER =",
    calculate_ber(
        bits,
        fsk_coherent_detected
    )
)

print(
    "FSK noncoherent BER =",
    calculate_ber(
        bits,
        fsk_noncoherent_detected
    )
)


samples_to_plot = min(
    5 * Ns,
    len(ask_tx)
)


# ASK waveform
plt.figure(figsize=(12, 5))

plt.plot(
    t_ask[:samples_to_plot] * 1000,
    ask_tx[:samples_to_plot]
)

plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.title("ASK Passband Waveform")

plt.grid()
plt.show()


# FSK waveform
plt.figure(figsize=(12, 5))

plt.plot(
    t_fsk[:samples_to_plot] * 1000,
    fsk_tx[:samples_to_plot]
)

plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.title("FSK Passband Waveform")

plt.grid()
plt.show()



freq_ask, power_ask = \
    power_spectrum(
        ask_tx,
        Fs
    )

plt.figure(figsize=(10, 5))

plt.plot(
    freq_ask / 1000,
    10 * np.log10(
        power_ask + 1e-12
    )
)

plt.xlabel("Frequency (kHz)")
plt.ylabel("Power (dB)")
plt.title("ASK Power Spectrum")

plt.grid()
plt.show()


freq_fsk, power_fsk = \
    power_spectrum(
        fsk_tx,
        Fs
    )

plt.figure(figsize=(10, 5))

plt.plot(
    freq_fsk / 1000,
    10 * np.log10(
        power_fsk + 1e-12
    )
)

plt.xlabel("Frequency (kHz)")
plt.ylabel("Power (dB)")
plt.title("FSK Power Spectrum")

plt.grid()
plt.show()



plt.figure(figsize=(10, 5))

plt.stem(
    range(len(bits)),
    ask_statistics
)

plt.axhline(
    0.25,
    linestyle='--',
    label="Decision threshold"
)

plt.xlabel("Bit index")
plt.ylabel("Correlation")
plt.title("ASK Coherent Correlator Output")

plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(10, 5))

plt.stem(
    range(len(bits)),
    corr0,
    label="f0 correlator"
)

plt.stem(
    range(len(bits)),
    corr1,
    label="f1 correlator"
)

plt.xlabel("Bit index")
plt.ylabel("Correlation")
plt.title("FSK Coherent Correlator Outputs")

plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(10, 5))

plt.stem(
    range(len(bits)),
    energy0,
    label="Energy at f0"
)

plt.stem(
    range(len(bits)),
    energy1,
    label="Energy at f1"
)

plt.xlabel("Bit index")
plt.ylabel("Energy")
plt.title("FSK Noncoherent Energy Detection")

plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))

plt.hist(
    ask_statistics[bits == 0],
    bins=10,
    alpha=0.7,
    label="Bit 0"
)

plt.hist(
    ask_statistics[bits == 1],
    bins=10,
    alpha=0.7,
    label="Bit 1"
)

plt.xlabel("Correlation statistic")
plt.ylabel("Number of bits")
plt.title("ASK Decision-Statistic Histogram")

plt.legend()
plt.grid()
plt.show()

SNR_values = np.arange(
    0,
    13,
    1
)

ber_ask = []
ber_fsk_coherent = []
ber_fsk_noncoherent = []


# Generate random bits for BER simulation
ber_bits = np.random.randint(
    0,
    2,
    Nbits_ber
)


for snr in SNR_values:

    tx, _ = ask_transmitter(
        ber_bits,
        fc_ask,
        Fs,
        Tb
    )

    rx = awgn(
        tx,
        snr
    )

    detected, _ = \
        ask_coherent_receiver(
            rx,
            fc_ask,
            Fs,
            Tb
        )

    ber_ask.append(
        calculate_ber(
            ber_bits,
            detected
        )
    )



    tx, _ = fsk_transmitter(
        ber_bits,
        f0,
        f1,
        Fs,
        Tb
    )

    rx = awgn(
        tx,
        snr
    )


    # Coherent FSK
    detected, _, _ = \
        fsk_coherent_receiver(
            rx,
            f0,
            f1,
            Fs,
            Tb
        )

    ber_fsk_coherent.append(
        calculate_ber(
            ber_bits,
            detected
        )
    )


    # Noncoherent FSK
    detected, _, _ = \
        fsk_noncoherent_receiver(
            rx,
            f0,
            f1,
            Fs,
            Tb
        )

    ber_fsk_noncoherent.append(
        calculate_ber(
            ber_bits,
            detected
        )
    )



plt.figure(figsize=(10, 6))

plt.semilogy(
    SNR_values,
    np.maximum(
        ber_ask,
        1e-5
    ),
    'o-',
    label="ASK coherent"
)

plt.semilogy(
    SNR_values,
    np.maximum(
        ber_fsk_coherent,
        1e-5
    ),
    's-',
    label="FSK coherent"
)

plt.semilogy(
    SNR_values,
    np.maximum(
        ber_fsk_noncoherent,
        1e-5
    ),
    '^-',
    label="FSK noncoherent"
)

plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title("BER vs SNR")

plt.grid()
plt.legend()
plt.show()

spacing_values = np.array([
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    3.0
])

spacing_ber = []

spacing_bits = np.random.randint(
    0,
    2,
    10000
)


for spacing in spacing_values:

    # Delta f = spacing / Tb
    test_delta_f = spacing / Tb

    test_f0 = f0
    test_f1 = f0 + test_delta_f


    tx, _ = fsk_transmitter(
        spacing_bits,
        test_f0,
        test_f1,
        Fs,
        Tb
    )


    rx = awgn(
        tx,
        snr_db
    )


    detected, _, _ = \
        fsk_coherent_receiver(
            rx,
            test_f0,
            test_f1,
            Fs,
            Tb
        )


    ber = calculate_ber(
        spacing_bits,
        detected
    )

    spacing_ber.append(ber)


plt.figure(figsize=(10, 6))

plt.semilogy(
    spacing_values,
    np.maximum(
        spacing_ber,
        1e-5
    ),
    'o-'
)

plt.axvline(
    1,
    linestyle='--',
    label=r"Minimum orthogonal spacing: $\Delta f T_b=1$"
)

plt.xlabel(r"Normalized tone spacing $\Delta f T_b$")
plt.ylabel("BER")
plt.title("FSK BER vs Tone Spacing")

plt.grid()
plt.legend()
plt.show()


print("\n============================================")
print("PROGRAM COMPLETED")
print("============================================")