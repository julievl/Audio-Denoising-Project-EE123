#!/usr/bin/env python
# coding: utf-8

# # Audio Signal Denoising using Different Filtering Methods

# ## Part 0: Setup and Helper Functions

# In[ ]:


# Part 0: Setup and Helper Functions

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import librosa.display
import IPython.display as ipd


# normalize audio so max amplitude close to 1
def normalize_audio(x):
    x = x.astype(float)
    return x / (np.max(np.abs(x)) + 1e-12)

# compute SNR between clean signal and test signal
def compute_snr(clean, test):
    error = test - clean
    return 10*np.log10(np.sum(clean**2) / (np.sum(error**2) + 1e-12))

# compute MSE between clean signal and test signal
def compute_mse(clean, test):
    return np.mean((clean - test)**2)

# plot magnitude spectrum
def plot_mag_spectrum(x, fs, n_fft = None, title = None, xlim = None):
    if n_fft is None:
        n_fft = int(2**np.ceil(np.log2(len(x))))

    X = np.fft.rfft(x, n = n_fft)
    f = np.fft.rfftfreq(n_fft, d = 1 / fs)
    mag_in_db = 20*np.log10(np.maximum(np.abs(X), 1e-12))

    plt.figure(figsize=(10,3))
    plt.plot(f, mag_in_db)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")

    if title: 
        plt.title(title)
    if xlim: 
        plt.xlim(xlim)

    plt.grid(True, alpha = 0.3)
    plt.show()

    return f, mag_in_db

# plot spectrogram using STFT
def show_spectrogram(x, fs, n_fft = 1024, hop_length = 256, title = None, fmax = None):
    S = librosa.stft(x, n_fft = n_fft, hop_length = hop_length, window = 'hann')
    S_in_db = librosa.amplitude_to_db(np.abs(S), ref = np.max)

    plt.figure(figsize=(10,4))
    librosa.display.specshow(S_in_db, sr = fs, hop_length = hop_length, x_axis = 'time', y_axis = 'hz')

    if fmax is not None:
        plt.ylim(0, fmax)

    plt.colorbar(format = '%+2.0f dB')

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.show()


# ## Part 1: Load Clean Audio

# In[ ]:


# Part 1: Load Clean Audio

clean_audio, fs = librosa.load("clean_speech.wav", sr = None, mono = True)
clean_audio = normalize_audio(clean_audio)

print("Sampling rate:", fs)
print("Length:", len(clean_audio)/fs, "seconds")
print()

print("Clean Audio:")
display(ipd.Audio(clean_audio, rate=fs))

plot_mag_spectrum(clean_audio, fs, title = "Magnitude Spectrum of Clean Audio", xlim = (0, 8000))
show_spectrogram(clean_audio, fs, title = "Spectrogram of Clean Audio", fmax = 8000)


# ## Part 2: Add Synthetic Noise

# In[ ]:


# Part 2: Add Synthetic Noise

target_snr_in_db = 5

# clean signal and noise to match target SNR
clean_signal_energy = np.mean(clean_audio**2)
noise_energy = clean_signal_energy / (10**(target_snr_in_db / 10))

# white Gaussian noise
white_noise = np.sqrt(noise_energy) * np.random.randn(len(clean_audio))

noisy_audio = white_noise + clean_audio

noisy_snr = compute_snr(clean_audio, noisy_audio)
noisy_mse = compute_mse(clean_audio, noisy_audio)

print("Target SNR: ", target_snr_in_db, "dB")
print("Noisy SNR: ", noisy_snr, "dB")
print("Noisy MSE: ", noisy_mse)
print()

print("Noisy Audio:")
display(ipd.Audio(noisy_audio, rate = fs))

plot_mag_spectrum(noisy_audio, fs, title = "Magnitude Spectrum of Noisy Audio", xlim = (0, 8000))
show_spectrogram(noisy_audio, fs, title = "Spectrogram of Noisy Audio", fmax = 8000)


# ## Part 3: Simple Filtering

# In[ ]:


# Part 3: Simple Filtering

# bandpass because audio is speech removes very low/high frequencies
# Butterworth bandpass filter, apply forward and backward to avoid phase distortion
def bandpass_filter(x, fs, cutoff_freq_low = 80, cutoff_freq_high = 4000, filter_order = 6):
    sos = signal.butter(filter_order, [cutoff_freq_low, cutoff_freq_high], btype = 'bandpass', fs = fs, output = 'sos')
    y = signal.sosfiltfilt(sos, x)
    return y


filtered_audio = bandpass_filter(noisy_audio, fs, cutoff_freq_low = 80, cutoff_freq_high = 4000, filter_order = 6)
filtered_snr = compute_snr(clean_audio, filtered_audio)
filtered_mse = compute_mse(clean_audio, filtered_audio)


print("Noisy SNR: ", noisy_snr, "dB")
print("Filtered SNR: ", filtered_snr, "dB")
print("SNR Improvement: ", filtered_snr - noisy_snr, "dB")
print()

print("Noisy MSE: ", noisy_mse)
print("Filtered MSE: ", filtered_mse)
print("MSE Reduction: ", noisy_mse - filtered_mse)
print()


print("Clean Audio:")
display(ipd.Audio(clean_audio, rate = fs))

print("Noisy Audio:")
display(ipd.Audio(noisy_audio, rate = fs))

print("Bandpass Filtered Audio:")
display(ipd.Audio(filtered_audio, rate = fs))
print()


plot_mag_spectrum(filtered_audio, fs, title = "Magnitude Spectrum of Filtered Audio", xlim = (0, 8000))
show_spectrogram(filtered_audio, fs, title = "Spectrogram of Filtered Audio", fmax = 8000)


# ## Part 4: Wiener Filtering

# In[ ]:


# Part 4.1: Wiener Filtering

# estimates cleaner signal using stats in a window
def wiener_filter(x, window_size = 501):
    y = signal.wiener(x, mysize = window_size)
    return y

# apply Wiener filter using initial window size
wiener_audio = wiener_filter(noisy_audio, window_size = 501)

wiener_snr = compute_snr(clean_audio, wiener_audio)
wiener_mse = compute_mse(clean_audio, wiener_audio)


print("Noisy SNR: ", noisy_snr, "dB")
print("Wiener SNR: ", wiener_snr, "dB")
print("SNR Improvement: ", wiener_snr - noisy_snr, "dB")
print()

print("Noisy MSE: ", noisy_mse)
print("Wiener MSE: ", wiener_mse)
print("MSE Reduction: ", noisy_mse - wiener_mse)
print()


print("Clean Audio:")
display(ipd.Audio(clean_audio, rate = fs))

print("Noisy Audio:")
display(ipd.Audio(noisy_audio, rate = fs))

print("Wiener Audio:")
display(ipd.Audio(wiener_audio, rate = fs))
print()


plot_mag_spectrum(wiener_audio, fs, title = "Magnitude Spectrum of Wiener Filtered Audio", xlim = (0, 8000))
show_spectrogram(wiener_audio, fs, title = "Spectrogram of Wiener Filtered Audio", fmax = 8000)


# In[ ]:


# Part 4.2: Comparing Different Window Sizes 

# test several window sizes
window_sizes = [51, 101, 251, 501, 1001, 2001]

print()
print(f"{'Window Size':<15}{'SNR [dB]':<15}{'  SNR Improvement [dB]':<25}{'          MSE':<15}{'              MSE Reduction':<15}")
print("-"*100)

# use best window size Wiener filter result
best_wiener_audio = None
best_wiener_snr = -np.inf
best_window_size = None

# apply Wiener with current window size, compute window metrics
for window_size in window_sizes:
    test = wiener_filter(noisy_audio, window_size = window_size)
    test_snr = compute_snr(clean_audio, test)
    test_mse = compute_mse(clean_audio, test)

    snr_improvement = test_snr - noisy_snr
    mse_reduction = noisy_mse - test_mse

    print(window_size, "\t  ", test_snr, "\t", snr_improvement, "\t", test_mse, "\t", mse_reduction)

    # update best result
    if test_snr > best_wiener_snr:
        best_wiener_snr = test_snr
        best_wiener_audio = test
        best_window_size = window_size

print()
print()
print("Best Wiener Window Size: \t", best_window_size)
print("Best Wiener SNR:\t        ", best_wiener_snr, "[dB]")



# In[ ]:


# Part 4.3: Best Window Size for Wiener Filter

# use output that gave best SNR
wiener_audio = best_wiener_audio
wiener_snr = compute_snr(clean_audio, wiener_audio)
wiener_mse = compute_mse(clean_audio, wiener_audio)


print("Noisy SNR: ", noisy_snr, "dB")
print("Wiener SNR: ", wiener_snr, "dB")
print("SNR Improvement: ", wiener_snr - noisy_snr, "dB")
print()

print("Noisy MSE: ", noisy_mse)
print("Wiener MSE: ", wiener_mse)
print("MSE Reduction: ", noisy_mse - wiener_mse)
print()


print("Clean Audio:")
display(ipd.Audio(clean_audio, rate = fs))

print("Noisy Audio:")
display(ipd.Audio(noisy_audio, rate = fs))

print("Wiener Audio:")
display(ipd.Audio(wiener_audio, rate = fs))
print()


plot_mag_spectrum(wiener_audio, fs, title = "Magnitude Spectrum of Best Wiener Filtered Audio", xlim = (0, 8000))
show_spectrogram(wiener_audio, fs, title = "Spectrogram of Best Wiener Filtered Audio", fmax = 8000)


# ## Part 5: Spectral Subtraction

# In[ ]:


# Part 5: Spectral Subtraction

# estimates noise spectrum subtracts from noisy speech spectrum
def spectral_subtraction(x, noise_estimation, n_fft = 1024, hop_length = 256, alpha = 1.0, beta = 0.02):
    # STFT of noisy signal
    X = librosa.stft(x, n_fft = n_fft, hop_length = hop_length, window = 'hann')
    X_mag = np.abs(X)
    X_phase = np.angle(X)

    # STFT of noise estimate
    N = librosa.stft(noise_estimation, n_fft = n_fft, hop_length = hop_length, window = 'hann')
    # average noise magnitude per frequency bin
    N_mag = np.mean(np.abs(N), axis = 1, keepdims = True)

    # subtract estimates noise magnitude without over subtracting
    clean_mag = np.maximum((X_mag - (alpha * N_mag)), (beta * X_mag))
    # reconstruct audio
    y = librosa.istft((clean_mag * np.exp(1j * X_phase)), hop_length = hop_length, length = len(x))
    return y


spectral_audio = spectral_subtraction(noisy_audio, white_noise, n_fft = 1024, hop_length = 256, alpha = 1.0, beta = 0.02)
spectral_snr =compute_snr(clean_audio, spectral_audio)
spectral_mse = compute_mse(clean_audio, spectral_audio)


print("Noisy SNR: ", noisy_snr, "dB")
print("Spectral Subtraction SNR: ", spectral_snr, "dB")
print("SNR Improvement: ", spectral_snr - noisy_snr, "dB")
print()

print("Noisy MSE: ", noisy_mse)
print("Spectral Subtraction MSE: ", spectral_mse)
print("MSE Reduction: ", noisy_mse - spectral_mse)
print()


print("Clean Audio:")
display(ipd.Audio(clean_audio, rate = fs))

print("Noisy Audio:")
display(ipd.Audio(noisy_audio, rate = fs))

print("Spectral Subtraction Audio:")
display(ipd.Audio(spectral_audio, rate = fs))
print()


plot_mag_spectrum(spectral_audio, fs, title = "Magnitude Spectrum of Spectral Subtraction Audio", xlim = (0, 8000))
show_spectrogram(spectral_audio, fs, title = "Spectrogram of Spectral Subtraction Audio", fmax = 8000)



# ## Part 6: Results

# In[ ]:


# Part 6: Results

# store each method
method = {"Noisy Audio": noisy_audio, "Bandpass Filter": filtered_audio, "Wiener Filter": wiener_audio, "Spectral Subtraction": spectral_audio}
print(f"{'Method':<15}{'         SNR [dB]':<15}{'        SNR Improvement [dB]':<25}{'          MSE':<15}{'        MSE Reduction':<15}")
print("-"*95)

# compute SNR/MSE for each method
for methods, audio in method.items():
    method_snr = compute_snr(clean_audio, audio)
    method_mse = compute_mse(clean_audio, audio)
    snr_improvement = method_snr - noisy_snr
    mse_reduction = noisy_mse - method_mse
    print(f"{methods:<25}{method_snr:<20.4f}{snr_improvement:<23.4f}{method_mse:<17.6f}{mse_reduction:<16.6f}")



print()
print("Clean Audio:")
display(ipd.Audio(clean_audio, rate = fs))
print()

print("Noisy Audio:")
display(ipd.Audio(noisy_audio, rate = fs))
print()

print("Bandpass Filtered Audio:")
display(ipd.Audio(filtered_audio, rate = fs))
print()

print("Wiener Filtered Audio:")
display(ipd.Audio(wiener_audio, rate = fs))
print()

print("Spectral Subtraction Audio:")
display(ipd.Audio(spectral_audio, rate = fs))
print()
print()



show_spectrogram(clean_audio, fs, title = "Spectrogram of Clean Audio", fmax = 8000)
show_spectrogram(noisy_audio, fs, title = "Spectrogram of Noisy Audio", fmax = 8000)
show_spectrogram(filtered_audio, fs, title = "Spectrogram of Bandpass Filtered Audio", fmax = 8000)
show_spectrogram(wiener_audio, fs, title = "Spectrogram of Best Wiener Filtered Audio", fmax = 8000)
show_spectrogram(spectral_audio, fs, title = "Spectrogram of Spectral Subtraction Audio", fmax = 8000)


# ## Part 7: Different Noise Types

# In[ ]:


# Part 7.1: Generating Different Noise Types For Comparison

# load and scale all noises and compute SNR/MSE for each
white_noise_audio = noisy_audio
target_power = clean_signal_energy / (10**(target_snr_in_db / 10))

fire_noise, fs_fire = librosa.load("fire_noise.wav", sr = fs, mono = True)
fire_noise = normalize_audio(fire_noise)

fire_noise = fire_noise[:len(clean_audio)]
fire_power = np.mean(fire_noise**2)
fire_noise_audio = clean_audio + (fire_noise * np.sqrt(target_power / fire_power))



cafe_noise, fs_cafe = librosa.load("noisy_cafe.wav", sr = fs, mono = True)
cafe_noise = normalize_audio(cafe_noise)

cafe_noise = cafe_noise[:len(clean_audio)]
cafe_power = np.mean(cafe_noise**2)
cafe_noise_audio = clean_audio + (cafe_noise * np.sqrt(target_power / cafe_power))



fire_noise_snr = compute_snr(clean_audio, fire_noise_audio)
fire_noise_mse = compute_mse(clean_audio, fire_noise_audio)

cafe_noise_snr = compute_snr(clean_audio, cafe_noise_audio)
cafe_noise_mse = compute_mse(clean_audio, cafe_noise_audio)



print()
print("Clean Audio:")
display(ipd.Audio(clean_audio, rate = fs))
plot_mag_spectrum(clean_audio, fs, title = "Magnitude Spectrum of Clean Audio", xlim = (0, 8000))
show_spectrogram(clean_audio, fs, title = "Spectrogram of Clean Audio", fmax = 8000)
print()

print("White Noise Audio:")
print("SNR:", noisy_snr, "dB")
print("MSE:", noisy_mse)
display(ipd.Audio(white_noise_audio, rate = fs))
plot_mag_spectrum(white_noise_audio, fs, title = "Magnitude Spectrum of White Noise Audio", xlim = (0, 8000))
show_spectrogram(white_noise_audio, fs, title = "Spectrogram of White Noise Audio", fmax = 8000)
print()

print("Fire Noise Audio:")
print("SNR:", fire_noise_snr, "dB")
print("MSE:", fire_noise_mse)
display(ipd.Audio(fire_noise_audio, rate = fs))
plot_mag_spectrum(fire_noise_audio, fs, title = "Magnitude Spectrum of Fire Noise Audio", xlim = (0, 8000))
show_spectrogram(fire_noise_audio, fs, title = "Spectrogram of Fire Noise Audio", fmax = 8000)
print()

print("Cafe Noise Audio:")
print("SNR:", cafe_noise_snr, "dB")
print("MSE:", cafe_noise_mse)
display(ipd.Audio(cafe_noise_audio, rate = fs))
plot_mag_spectrum(cafe_noise_audio, fs, title = "Magnitude Spectrum of Cafe Noise Audio", xlim = (0, 8000))
show_spectrogram(cafe_noise_audio, fs, title = "Spectrogram of Cafe Noise Audio", fmax = 8000)
print()




# In[ ]:


# Part 7.2: Finding Best Filter for Additional Noises

# apply all three denoising methods to each noise type

# Fire Noise

fire_filtered_audio = bandpass_filter(fire_noise_audio, fs, cutoff_freq_low = 80, cutoff_freq_high = 4000, filter_order = 6)
fire_filtered_snr = compute_snr(clean_audio, fire_filtered_audio)
fire_filtered_mse = compute_mse(clean_audio, fire_filtered_audio)

fire_wiener_audio = wiener_filter(fire_noise_audio, window_size = best_window_size)
fire_wiener_snr = compute_snr(clean_audio, fire_wiener_audio)
fire_wiener_mse = compute_mse(clean_audio, fire_wiener_audio)

fire_spectral_audio = spectral_subtraction(fire_noise_audio, fire_noise_audio - clean_audio, n_fft = 1024, hop_length = 256, alpha = 1.0, beta = 0.02)
fire_spectral_snr = compute_snr(clean_audio, fire_spectral_audio)
fire_spectral_mse = compute_mse(clean_audio, fire_spectral_audio)

print("Fire Noise:")
print("Noisy SNR: ", fire_noise_snr, "dB")
print("Noisy MSE: ", fire_noise_mse)
print()

print("Bandpass SNR: ", fire_filtered_snr, "dB")
print("Bandpass SNR Improvement: ", fire_filtered_snr - fire_noise_snr, "dB")
print("Bandpass MSE: ", fire_filtered_mse)
print("Bandpass MSE Reduction: ", fire_noise_mse - fire_filtered_mse)
print()

print("Wiener SNR: ", fire_wiener_snr, "dB")
print("Wiener SNR Improvement: ", fire_wiener_snr - fire_noise_snr, "dB")
print("Wiener MSE: ", fire_wiener_mse)
print("Wiener MSE Reduction: ", fire_noise_mse - fire_wiener_mse)
print()

print("Spectral Subtraction SNR: ", fire_spectral_snr, "dB")
print("Spectral Subtraction SNR Improvement: ", fire_spectral_snr - fire_noise_snr, "dB")
print("Spectral Subtraction MSE: ", fire_spectral_mse)
print("Spectral Subtraction MSE Reduction: ", fire_noise_mse - fire_spectral_mse)
print()


print("Clean Audio:")
display(ipd.Audio(clean_audio, rate = fs))

print("Fire Noise Bandpass Filtered Audio:")
display(ipd.Audio(fire_filtered_audio, rate = fs))
print()

print("Fire Noise Wiener Filtered Audio:")
display(ipd.Audio(fire_wiener_audio, rate = fs))
print()

print("Fire Noise Spectral Subtraction Filtered Audio:")
display(ipd.Audio(fire_spectral_audio, rate = fs))
print()


plot_mag_spectrum(fire_filtered_audio, fs, title = "Magnitude Spectrum of Fire Noise Bandpass Filtered Audio", xlim = (0, 8000))
show_spectrogram(fire_filtered_audio, fs, title = "Spectrogram of Fire Noise Bandpass Filtered Audio", fmax = 8000)

plot_mag_spectrum(fire_wiener_audio, fs, title = "Magnitude Spectrum of Fire Noise Wiener Filtered Audio", xlim = (0, 8000))
show_spectrogram(fire_wiener_audio, fs, title = "Spectrogram of Fire Noise Wiener Filtered Audio", fmax = 8000)

plot_mag_spectrum(fire_spectral_audio, fs, title = "Magnitude Spectrum of Fire Noise Spectral Subtraction Filtered Audio", xlim = (0, 8000))
show_spectrogram(fire_spectral_audio, fs, title = "Spectrogram of Fire Noise Spectral Subtraction Filtered Audio", fmax = 8000)

# Cafe Noise

cafe_filtered_audio = bandpass_filter(cafe_noise_audio, fs, cutoff_freq_low = 80, cutoff_freq_high = 4000, filter_order = 6)
cafe_filtered_snr = compute_snr(clean_audio, cafe_filtered_audio)
cafe_filtered_mse = compute_mse(clean_audio, cafe_filtered_audio)

cafe_wiener_audio = wiener_filter(cafe_noise_audio, window_size = best_window_size)
cafe_wiener_snr = compute_snr(clean_audio, cafe_wiener_audio)
cafe_wiener_mse = compute_mse(clean_audio, cafe_wiener_audio)

cafe_spectral_audio = spectral_subtraction(cafe_noise_audio, cafe_noise_audio - clean_audio, n_fft = 1024, hop_length = 256, alpha = 1.0, beta = 0.02)
cafe_spectral_snr = compute_snr(clean_audio, cafe_spectral_audio)
cafe_spectral_mse = compute_mse(clean_audio, cafe_spectral_audio)

print("Cafe Noise:")
print("Noisy SNR: ", cafe_noise_snr, "dB")
print("Noisy MSE: ", cafe_noise_mse)
print()

print("Bandpass SNR: ", cafe_filtered_snr, "dB")
print("Bandpass SNR Improvement: ", cafe_filtered_snr - cafe_noise_snr, "dB")
print("Bandpass MSE: ", cafe_filtered_mse)
print("Bandpass MSE Reduction: ", cafe_noise_mse - cafe_filtered_mse)
print()

print("Wiener SNR: ", cafe_wiener_snr, "dB")
print("Wiener SNR Improvement: ", cafe_wiener_snr - cafe_noise_snr, "dB")
print("Wiener MSE: ", cafe_wiener_mse)
print("Wiener MSE Reduction: ", cafe_noise_mse - cafe_wiener_mse)
print()

print("Spectral Subtraction SNR: ", cafe_spectral_snr, "dB")
print("Spectral Subtraction SNR Improvement: ", cafe_spectral_snr - cafe_noise_snr, "dB")
print("Spectral Subtraction MSE: ", cafe_spectral_mse)
print("Spectral Subtraction MSE Reduction: ", cafe_noise_mse - cafe_spectral_mse)
print()


print("Clean Audio:")
display(ipd.Audio(clean_audio, rate = fs))

print("Cafe Noise Bandpass Filtered Audio:")
display(ipd.Audio(cafe_filtered_audio, rate = fs))
print()

print("Cafe Noise Wiener Filtered Audio:")
display(ipd.Audio(cafe_wiener_audio, rate = fs))
print()

print("Cafe Noise Spectral Subtraction Filtered Audio:")
display(ipd.Audio(cafe_spectral_audio, rate = fs))
print()


plot_mag_spectrum(cafe_filtered_audio, fs, title = "Magnitude Spectrum of Cafe Noise Bandpass Filtered Audio", xlim = (0, 8000))
show_spectrogram(cafe_filtered_audio, fs, title = "Spectrogram of Cafe Noise Bandpass Filtered Audio", fmax = 8000)

plot_mag_spectrum(cafe_wiener_audio, fs, title = "Magnitude Spectrum of Cafe Noise Wiener Filtered Audio", xlim = (0, 8000))
show_spectrogram(cafe_wiener_audio, fs, title = "Spectrogram of Cafe Noise Wiener Filtered Audio", fmax = 8000)

plot_mag_spectrum(cafe_spectral_audio, fs, title = "Magnitude Spectrum of Cafe Noise Spectral Subtraction Filtered Audio", xlim = (0, 8000))
show_spectrogram(cafe_spectral_audio, fs, title = "Spectrogram of Cafe Noise Spectral Subtraction Filtered Audio", fmax = 8000)


# In[ ]:


# Part 7.3: Display Best Filter for each Noise

# White Noise

print()
print("White Noise:")
print("Noisy SNR: ", noisy_snr, "dB")
print("Noisy MSE: ", noisy_mse)

print("Bandpass SNR: ", filtered_snr, "dB")
print("Wiener SNR: ", wiener_snr, "dB")
print("Spectral Subtraction SNR: ", spectral_snr, "dB")
print("Spectral Subtraction MSE: ", spectral_mse)
print()


print("Clean Audio:")
display(ipd.Audio(clean_audio, rate = fs))

print("White Noise Audio:")
display(ipd.Audio(noisy_audio, rate = fs))

print("Best Method for White Noise: Spectral Subtraction")
print("Best SNR Improvement: ", spectral_snr - noisy_snr, "dB")
print("Best MSE Reduction: ", noisy_mse - spectral_mse)
print()
print("Spectral Subtraction Filtered White Noise Audio:")
display(ipd.Audio(spectral_audio, rate = fs))
print()

plot_mag_spectrum(spectral_audio, fs, title = "Magnitude Spectrum of Spectral Subtraction Audio for White Noise", xlim = (0, 8000))
show_spectrogram(spectral_audio, fs, title = "Spectrogram of Spectral Subtraction Audio for White Noise", fmax = 8000)

# Fire Noise

print()
print("Fire Noise:")
print("Noisy SNR: ", fire_noise_snr, "dB")
print("Noisy MSE: ", fire_noise_mse)

print("Bandpass SNR: ", fire_filtered_snr, "dB")
print("Wiener SNR: ", fire_wiener_snr, "dB")
print("Spectral Subtraction SNR: ", fire_spectral_snr, "dB")
print("Spectral Subtraction MSE: ", fire_spectral_mse)
print()


print("Clean Audio:")
display(ipd.Audio(clean_audio, rate = fs))

print("Fire Noise Audio:")
display(ipd.Audio(fire_noise_audio, rate = fs))

print("Best Method for Fire Noise: Spectral Subtraction")
print("Best SNR Improvement: ", fire_spectral_snr - fire_noise_snr, "dB")
print("Best MSE Reduction: ", fire_noise_mse - fire_spectral_mse)
print()
print("Spectral Subtraction Filtered Fire Noise Audio:")
display(ipd.Audio(fire_spectral_audio, rate = fs))
print()

plot_mag_spectrum(fire_spectral_audio, fs, title = "Magnitude Spectrum of Spectral Subtraction Audio for Fire Noise", xlim = (0, 8000))
show_spectrogram(fire_spectral_audio, fs, title = "Spectrogram of Spectral Subtraction Audio for Fire Noise", fmax = 8000)

# Cafe Noise

print()
print("Cafe Noise:")
print("Noisy SNR: ", cafe_noise_snr, "dB")
print("Noisy MSE: ", cafe_noise_mse)

print("Bandpass SNR: ", cafe_filtered_snr, "dB")
print("Wiener SNR: ", cafe_wiener_snr, "dB")
print("Spectral Subtraction SNR: ", cafe_spectral_snr, "dB")
print("Spectral Subtraction MSE: ", cafe_spectral_mse)
print()


print("Clean Audio:")
display(ipd.Audio(clean_audio, rate = fs))

print("Cafe Noise Audio:")
display(ipd.Audio(cafe_noise_audio, rate = fs))

print("Best Method for Cafe Noise: Spectral Subtraction")
print("Best SNR Improvement: ", cafe_spectral_snr - cafe_noise_snr, "dB")
print("Best MSE Reduction: ", cafe_noise_mse - cafe_spectral_mse)
print()
print("Spectral Subtraction Filtered Cafe Noise Audio:")
display(ipd.Audio(cafe_spectral_audio, rate = fs))
print()

plot_mag_spectrum(cafe_spectral_audio, fs, title = "Magnitude Spectrum of Spectral Subtraction Audio for Cafe Noise", xlim = (0, 8000))
show_spectrogram(cafe_spectral_audio, fs, title = "Spectrogram of Spectral Subtraction Audio for Cafe Noise", fmax = 8000)


# In[ ]:


import soundfile as sf

# demo clips for presentation
sf.write("clean_audio_demo.wav", clean_audio, fs)
sf.write("cafe_noisy_audio_demo.wav", cafe_noise_audio, fs)
sf.write("cafe_bandpass_demo.wav", cafe_filtered_audio, fs)
sf.write("cafe_wiener_demo.wav", cafe_wiener_audio, fs)
sf.write("cafe_spectral_subtraction_demo.wav", cafe_spectral_audio, fs)


# In[ ]:


# zip files for gradescope

import zipfile
with zipfile.ZipFile("audio_denoising.zip", "w") as zipf:
    zipf.write("Audio_Denoising_Project.ipynb")
    zipf.write("clean_speech.wav")
    zipf.write("fire_noise.wav")
    zipf.write("noisy_cafe.wav")

