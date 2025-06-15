import os
import torchaudio

def print_sample_rates(folder_path):
    """
    Muestra el sample rate de cada archivo .wav en la carpeta dada.
    """
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(".wav"):
            filepath = os.path.join(folder_path, filename)
            try:
                _, sr = torchaudio.load(filepath)
                print(f"{filename}: {sr} Hz")
            except Exception as e:
                print(f"Error leyendo {filename}: {e}")


print_sample_rates("audios/inputs")




# set sample rate to 44100Hz
from torchaudio.functional import resample

def resample_to_44100(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in sorted(os.listdir(input_folder)):
        if not filename.lower().endswith(".wav"):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            waveform, sr = torchaudio.load(input_path)
            if sr != 44100:
                print(f"{filename}: {sr} Hz → 44100 Hz")
                waveform = resample(waveform, orig_freq=sr, new_freq=44100)
            else:
                print(f"{filename}: ya está a 44100 Hz")

            torchaudio.save(output_path, waveform, 44100)

        except Exception as e:
            print(f"Error con {filename}: {e}")


resample_to_44100("audios/inputs", "audios/inp_44100")
