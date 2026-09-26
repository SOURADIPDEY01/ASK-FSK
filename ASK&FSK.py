import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# ASK AND BFSK TRANSMITTER / CHANNEL / RECEIVER
# INPUT BASED PROGRAM
# ============================================================

np.random.seed(10)

# ============================================================
# USER INPUT
# ============================================================

bits_input = input("Enter binary data (e.g. 10110010): ").strip()

# Check binary input
if not all(bit in '01' for bit in bits_input):
    raise ValueError("Enter only 0 and 1.")

bits = np.array([int(bit) for bit in bits_input])

Rb = float(input("Enter bit rate Rb (bits/sec): "))
Fs = float(input("Enter sampling frequency Fs (Hz): "))

fc_ask = float(input("Enter ASK carrier frequency fc (Hz): "))

f0 = float(input("Enter BFSK frequency f0 for bit 0 (Hz): "))
f1 = float(input("Enter BFSK frequency f1 for bit 1 (Hz): "))

snr_db = float(input("Enter SNR (dB): "))

# BER simulation parameters
Nbits_ber = int(input("Enter number of bits for BER simulation: "))

# ------------------------------------------------------------
# Basic parameters
# ------------------------------------------------------------

Tb = 1 / Rb
Ns = int(Fs * Tb)

if Ns < 2:
    raise ValueError("Sampling frequency must be sufficiently high.")

print("\n==========================================")
print("SYSTEM PARAMETERS")
print("==========================================")
print("Input bits       :", bits_input)
print("Number of bits   :", len(bits))
print("Bit rate         :", Rb, "bits/sec")
print("Bit duration     :", Tb, "sec")
print("Sampling freq.   :", Fs, "Hz")
print("ASK carrier      :", fc_ask, "Hz")
print("BFSK f0          :", f0, "Hz")
print("BFSK f1          :", f1, "Hz")
print("Tone spacing     :", abs(f1 - f0), "Hz")
print("SNR              :", snr_db, "dB")

print("\nFSK orthogonality condition:")
print("Delta_f = k/Tb")
print("Minimum Delta_f =", 1/Tb, "Hz")

if abs(abs(f1 - f0) * Tb - round(abs(f1 - f0) * Tb)) < 1e-10:
    print("The selected BFSK tones are orthogonal.")
else:
    print("The selected BFSK tones are NOT orthogonal.")


# ============================================================
# ASK TRANSMITTER
# ============================================================

def ask_transmitter(bits, fc, Fs, Tb):

    Ns = int(Fs * Tb)

    t_bit = np.arange(Ns) / Fs

    carrier = np.cos(2 * np.pi * fc * t_bit)

    signal = np.concatenate([
        carrier if bit == 1 else np.zeros(Ns)
        for bit in bits
    ])

    t = np.arange(len(signal)) / Fs

    return signal, t


# ============================================================
# BFSK TRANSMITTER
# ============================================================

def bfsk_transmitter(bits, f0, f1, Fs, Tb):

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

    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power) * np.random.randn(
        len(signal)
    )

    return signal + noise


# ============================================================
# ASK COHERENT RECEIVER
# ============================================================

def ask_coherent_receiver(rx, fc, Fs, Tb):

    Ns = int(Fs * Tb)

    Nbits = len(rx) // Ns

    t_bit = np.arange(Ns) / Fs

    reference = np.cos(
        2 * np.pi * fc * t_bit
    )

    statistics = []

    for i in range(Nbits):

        r = rx[i * Ns:(i + 1) * Ns]

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
# BFSK COHERENT RECEIVER
# ============================================================

def bfsk_coherent_receiver(
    rx, f0, f1, Fs, Tb
):

    Ns = int(Fs * Tb)

    Nbits = len(rx) // Ns

    t_bit = np.arange(Ns) / Fs

    ref0 = np.cos(
        2 * np.pi * f0 * t_bit
    )

    ref1 = np.cos(
        2 * np.pi * f1 * t_bit
    )

    corr0 = []
    corr1 = []
    detected_bits = []

    for i in range(Nbits):

        r = rx[i * Ns:(i + 1) * Ns]

        z0 = np.sum(
            r * ref0
        ) / Ns

        z1 = np.sum(
            r * ref1
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


# ============================================================
# BFSK NONCOHERENT ENERGY DETECTOR
# ============================================================

def bfsk_noncoherent_receiver(
    rx, f0, f1, Fs, Tb
):

    Ns = int(Fs * Tb)

    Nbits = len(rx) // Ns

    t_bit = np.arange(Ns) / Fs

    # f0 reference signals
    f0_cos = np.cos(
        2 * np.pi * f0 * t_bit
    )

    f0_sin = np.sin(
        2 * np.pi * f0 * t_bit
    )

    # f1 reference signals
    f1_cos = np.cos(
        2 * np.pi * f1 * t_bit
    )

    f1_sin = np.sin(
        2 * np.pi * f1 * t_bit
    )

    energy0 = []
    energy1 = []
    detected_bits = []

    for i in range(Nbits):

        r = rx[i * Ns:(i + 1) * Ns]

        I0 = np.sum(
            r * f0_cos
        ) / Ns

        Q0 = np.sum(
            r * f0_sin
        ) / Ns

        I1 = np.sum(
            r * f1_cos
        ) / Ns

        Q1 = np.sum(
            r * f1_sin
        ) / Ns

        E0 = I0**2 + Q0**2
        E1 = I1**2 + Q1**2

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

    freq = np.fft.fftfreq(
        N,
        1 / Fs
    )

    positive = freq >= 0

    freq = freq[positive]

    power = (
        np.abs(X[positive])**2 / N
    )

    return freq, power


# ============================================================
# BER
# ============================================================

def calculate_ber(original, detected):

    return np.mean(
        original != detected
    )


# ============================================================
# TRANSMITTER
# ============================================================

ask_tx, t_ask = ask_transmitter(
    bits,
    fc_ask,
    Fs,
    Tb
)

bfsk_tx, t_bfsk = bfsk_transmitter(
    bits,
    f0,
    f1,
    Fs,
    Tb
)


# ============================================================
# CHANNEL
# ============================================================

ask_rx = awgn(
    ask_tx,
    snr_db
)

bfsk_rx = awgn(
    bfsk_tx,
    snr_db
)


# ============================================================
# RECEIVERS
# ============================================================

ask_detected, ask_stats = \
    ask_coherent_receiver(
        ask_rx,
        fc_ask,
        Fs,
        Tb
    )

(
    bfsk_coherent_detected,
    corr0,
    corr1
) = bfsk_coherent_receiver(
    bfsk_rx,
    f0,
    f1,
    Fs,
    Tb
)

(
    bfsk_noncoherent_detected,
    energy0,
    energy1
) = bfsk_noncoherent_receiver(
    bfsk_rx,
    f0,
    f1,
    Fs,
    Tb
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n==========================================")
print("TRANSMISSION RESULTS")
print("==========================================")

print("\nOriginal bits:")
print(bits_input)

print("\nASK coherent detected:")
print(''.join(
    map(str, ask_detected)
))

print("\nBFSK coherent detected:")
print(''.join(
    map(str, bfsk_coherent_detected)
))

print("\nBFSK noncoherent detected:")
print(''.join(
    map(str, bfsk_noncoherent_detected)
))

print("\n------------------------------------------")

print("ASK coherent BER:",
      calculate_ber(bits, ask_detected))

print("BFSK coherent BER:",
      calculate_ber(
          bits,
          bfsk_coherent_detected
      ))

print("BFSK noncoherent BER:",
      calculate_ber(
          bits,
          bfsk_noncoherent_detected
      ))


# ============================================================
# PASSBAND WAVEFORMS
# ============================================================

samples_to_plot = min(
    5 * Ns,
    len(ask_tx)
)

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


plt.figure(figsize=(12, 5))

plt.plot(
    t_bfsk[:samples_to_plot] * 1000,
    bfsk_tx[:samples_to_plot]
)

plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.title("BFSK Passband Waveform")
plt.grid()

plt.show()


# ============================================================
# POWER SPECTRA
# ============================================================

freq, power = power_spectrum(
    ask_tx,
    Fs
)

plt.figure(figsize=(10, 5))

plt.plot(
    freq / 1000,
    10 * np.log10(power + 1e-12)
)

plt.xlabel("Frequency (kHz)")
plt.ylabel("Power (dB)")
plt.title("ASK Power Spectrum")
plt.grid()

plt.show()


freq, power = power_spectrum(
    bfsk_tx,
    Fs
)

plt.figure(figsize=(10, 5))

plt.plot(
    freq / 1000,
    10 * np.log10(power + 1e-12)
)

plt.xlabel("Frequency (kHz)")
plt.ylabel("Power (dB)")
plt.title("BFSK Power Spectrum")
plt.grid()

plt.show()


# ============================================================
# ASK CORRELATOR OUTPUT
# ============================================================

plt.figure(figsize=(10, 5))

plt.stem(
    range(len(bits)),
    ask_stats
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


# ============================================================
# BFSK CORRELATOR OUTPUT
# ============================================================

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
plt.title("BFSK Coherent Correlator Outputs")

plt.legend()
plt.grid()

plt.show()


# ============================================================
# BFSK ENERGY DETECTOR
# ============================================================

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
plt.title("BFSK Noncoherent Energy Detection")

plt.legend()
plt.grid()

plt.show()


# ============================================================
# DECISION STATISTIC HISTOGRAM
# ============================================================

plt.figure(figsize=(10, 5))

plt.hist(
    ask_stats[bits == 0],
    bins=10,
    alpha=0.7,
    label="Bit 0"
)

plt.hist(
    ask_stats[bits == 1],
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


# ============================================================
# BER vs SNR
# ============================================================

SNR_values = np.arange(
    0,
    13,
    1
)

ber_ask = []
ber_bfsk_coherent = []
ber_bfsk_noncoherent = []

ber_bits = np.random.randint(
    0,
    2,
    Nbits_ber
)

for snr in SNR_values:

    # ---------------- ASK ----------------

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

    # ---------------- BFSK ----------------

    tx, _ = bfsk_transmitter(
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

    # Coherent
    detected, _, _ = \
        bfsk_coherent_receiver(
            rx,
            f0,
            f1,
            Fs,
            Tb
        )

    ber_bfsk_coherent.append(
        calculate_ber(
            ber_bits,
            detected
        )
    )

    # Noncoherent
    detected, _, _ = \
        bfsk_noncoherent_receiver(
            rx,
            f0,
            f1,
            Fs,
            Tb
        )

    ber_bfsk_noncoherent.append(
        calculate_ber(
            ber_bits,
            detected
        )
    )


# ============================================================
# BER vs SNR PLOT
# ============================================================

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
        ber_bfsk_coherent,
        1e-5
    ),
    's-',
    label="BFSK coherent"
)

plt.semilogy(
    SNR_values,
    np.maximum(
        ber_bfsk_noncoherent,
        1e-5
    ),
    '^-',
    label="BFSK noncoherent"
)

plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title("BER vs SNR")

plt.grid()
plt.legend()

plt.show()


# ============================================================
# BER vs TONE SPACING
# ============================================================

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

spacing_snr = snr_db

for spacing in spacing_values:

    delta_f = spacing / Tb

    test_f0 = f0
    test_f1 = f0 + delta_f

    tx, _ = bfsk_transmitter(
        spacing_bits,
        test_f0,
        test_f1,
        Fs,
        Tb
    )

    rx = awgn(
        tx,
        spacing_snr
    )

    detected, _, _ = \
        bfsk_coherent_receiver(
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


# ============================================================
# BER vs TONE SPACING PLOT
# ============================================================

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
    label="Minimum orthogonal spacing"
)

plt.xlabel(r"$\Delta f T_b$")
plt.ylabel("BER")
plt.title("BFSK BER vs Tone Spacing")

plt.grid()
plt.legend()

plt.show()