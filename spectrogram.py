import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def save_spectrogram(wav_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    plt.figure(figsize=(10, 4))
    plt.specgram(audio, NFFT=1024, Fs=sr, noverlap=512, cmap='inferno')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label='dB')
    plt.title(f"Spectrogram original\n{os.path.basename(wav_path)}")
    plt.tight_layout()
    output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(wav_path))[0] + "_original.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Guardado: {output_path}")

def save_spectrograms_from_folder(folder, output_folder="spectrograms"):
    for fname in os.listdir(folder):
        if fname.endswith(".wav"):
            save_spectrogram(os.path.join(folder, fname), output_folder)


save_spectrogram("data/input_zeropadding/36.wav","material_memoria/specs")


def save_spectrogram(wav_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    plt.figure(figsize=(10, 4))
    plt.specgram(audio, NFFT=1024, Fs=sr, noverlap=512, cmap='inferno')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label='dB')
    plt.title(f"Pedalboard Spectrogram\n{os.path.basename(wav_path)}")
    plt.tight_layout()
    output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(wav_path))[0] + ".png")
    plt.savefig(output_path)
    plt.close()
    print(f"Guardado: {output_path}")

def save_spectrograms_from_folder(folder, output_folder="spectrograms"):
    for fname in os.listdir(folder):
        if fname.endswith(".wav"):
            save_spectrogram(os.path.join(folder, fname), output_folder)


#save_spectrogram("results_pedalboard/pedal_norm_peak/36_pedal_norm_peak.wav","specs")
def save_waveform(wav_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    plt.figure(figsize=(16, 4))  # Más ancho para más detalle temporal
    times = np.arange(len(audio)) / sr
    plt.plot(times, audio, color='royalblue', linewidth=0.7, alpha=0.9)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(f"Waveform\n{os.path.basename(wav_path)}")
    plt.ylim(-1, 1)  # Escala fija de amplitud
    plt.xlim(0, len(audio) / sr)  # Todo el rango temporal
    plt.tight_layout()
    base = os.path.splitext(os.path.basename(wav_path))[0]
    output_path = os.path.join(output_folder, base + "_peak.png")
    plt.savefig(output_path, dpi=300)  # Alta resolución horizontal
    plt.close()
    print(f"Saved: {output_path}")

#save_waveform("results_pedalboard/pedal_norm_peak/23_pedal_norm_peak.wav","material_memoria/waveforms")


def save_waveform(wav_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    plt.figure(figsize=(16, 4))
    times = np.arange(len(audio)) / sr
    plt.plot(times, audio, color='royalblue', linewidth=0.7, alpha=0.9)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(f"Waveform\n{os.path.basename(wav_path)}")
    # Zoom automático en el eje y para señales de bajo nivel
    min_amp = np.percentile(audio, 0.5)
    max_amp = np.percentile(audio, 99.5)
    if min_amp == max_amp:
        min_amp, max_amp = audio.min(), audio.max()
    plt.ylim(min_amp, max_amp)
    plt.xlim(0, len(audio) / sr)
    plt.tight_layout()
    base = os.path.splitext(os.path.basename(wav_path))[0]
    output_path = os.path.join(output_folder, base + "_pedalboard_no_normalization_waveform.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

#save_waveform("results_pedalboard/pedal_no_norm/36.wav","material_memoria/waveforms")
save_waveform("results_pedalboard/pedal_no_norm/23.wav","material_memoria/waveforms")