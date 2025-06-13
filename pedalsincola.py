import numpy as np
import soundfile as sf
from pedalboard import Pedalboard, Convolution
from scipy import signal
import matplotlib.pyplot as plt
import json
from tqdm import tqdm


def pedalboard_convolution_no_tail(input_wav: str, ir_wav: str, output_wav: str) -> None:
    """
    Aplica convolución de reverb con Pedalboard y guarda solo la señal procesada,
    sin añadir manualmente la cola (tail) con buffer de ceros.
    """
    x, sr_x = sf.read(input_wav)
    if x.ndim > 1:
        x = x[:, 0]
    board = Pedalboard([Convolution(ir_wav)])
    y = board.process(x, sr_x)
    sf.write(output_wav, y, sr_x)
    print(f"Guardado: {output_wav}")


#pedalboard_convolution_no_tail("data/original/1.wav","IRs/Studio Nord Spring 3,0 flat.wav","output_pedalboard_no_tail.wav")


import os
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

import os
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def plot_duration_histogram_professional(wav_folder, output_folder="hist_plots", output_png="durations_histogram.png"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    durations = []
    for fname in sorted(os.listdir(wav_folder)):
        if fname.lower().endswith('.wav'):
            path = os.path.join(wav_folder, fname)
            audio, sr = sf.read(path)
            duration = round(len(audio) / sr, 2)  # Redondea a 2 decimales
            durations.append(duration)
    # Cuenta ocurrencias de cada duración
    duration_counts = Counter(durations)
    # Ordena por duración
    sorted_durations = sorted(duration_counts.items())
    x = [str(d) for d, _ in sorted_durations]
    y = [c for _, c in sorted_durations]

    plt.figure(figsize=(14, 6))
    bars = plt.bar(x, y, color='orange', edgecolor='black', alpha=0.85)
    plt.xlabel('Duration (seconds)', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.title('Histogram of WAV Durations (Grouped)', fontsize=18)
    plt.xticks(rotation=45, fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    output_path = os.path.join(output_folder, output_png)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

# Ejemplo de uso:
plot_duration_histogram_professional("data/input_zeropadding", output_folder="data/input_zeropadding")
