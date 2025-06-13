import gdown
import librosa
import librosa.display
import numpy as np
import soundfile as sf
from statsmodels.tsa.statespace.sarimax import SARIMAX

i = 0

# download file from EGDB 
def download_audio(file_url, output_path=f"data/original/downloaded_audio{i}.wav"):
    gdown.download(file_url, output_path, quiet=False)
    return output_path

# load the file
def load_audio(file_path, sr=22050):
    audio, sr = librosa.load(file_path, sr=sr, mono=True)
    return audio, sr

# fit farina model and generate synthetic reverb
def apply_farina_convolution(audio, order=(1,0,1), steps=500):
    model = SARIMAX(audio, order=order)
    model_fit = model.fit(disp=False)
    synthetic_reverb = model_fit.predict(start=0, end=len(audio) + steps)
    return synthetic_reverb

# save new synthetic audio file
def save_audio(file_path, audio, sr):
    sf.write(file_path, audio, sr)

if __name__ == "__main__":
    # audio_url = "https://drive.google.com/uc?id={FILE_ID}" (replace with actual EGDB file link)
    audio_url = "https://drive.google.com/uc?id=1AT6sJw5FZ7LqesXDPS8754NUA0r9HV53"

    # we download and process the audio
    file_path = download_audio(audio_url)
    audio, sr = load_audio(file_path)
    
    # farina convolution
    # synthetic_audio = apply_farina_convolution(audio, (2,1,2))
    synthetic_audio = apply_farina_convolution(audio)
    
    # save the generated audio
    save_audio(f"data/synth/synthetic_reverb{i}.wav", synthetic_audio, sr)
    print(f"Synthetic audio saved as 'synthetic_reverb{i}.wav'")
