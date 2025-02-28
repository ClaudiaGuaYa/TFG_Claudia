import os
import numpy as np
import soundfile as sf
import scipy.signal as signal
import json
from tqdm import tqdm

def extract_ir_parameters(ir_wav):
   
    ir_signal, ir_sr = sf.read(ir_wav)
    if ir_signal.ndim > 1:
        ir_signal = ir_signal[:, 0]
    
    energy = np.cumsum(ir_signal[::-1]**2)[::-1]
    decay_time = np.argmax(energy < max(energy) * 0.001) / ir_sr
    
    spectrum = np.fft.rfft(ir_signal)
    spectral_centroid = np.sum(np.abs(spectrum) * np.arange(len(spectrum))) / np.sum(np.abs(spectrum))
    
    params = {
        "sampling_rate": ir_sr,
        "decay_time": decay_time,
        "spectral_centroid": spectral_centroid
    }
    
    print("IR params:", params)  #add delay parameter
    return params


def partitioned_fft_convolution(input_wav, ir_wav, output_wav, partition_size=1024):
   
    input_signal, input_sr = sf.read(input_wav)
    ir_signal, ir_sr = sf.read(ir_wav)
    
    # sampling rates match
    if input_sr != ir_sr:
        raise ValueError(f"Sampling rates do not match: {input_sr} vs {ir_sr}")
    
    # IR to mono if it's multi-channel
    if ir_signal.ndim > 1:
        ir_signal = ir_signal[:, 0]
    
    remainder = len(ir_signal) % partition_size
    pad_width = partition_size - remainder if remainder != 0 else 0
    ir_padded = np.pad(ir_signal, (0, pad_width), mode='constant')
    
    # IR partitions (FFT of each partition)
    num_partitions = len(ir_padded) // partition_size
    ir_partitions = np.array([
        np.fft.rfft(ir_padded[i * partition_size:(i + 1) * partition_size])
        for i in range(num_partitions)
    ])
    
    output_length = len(input_signal) + len(ir_signal) - 1
    output_signal = np.zeros(output_length)
    

    for i in tqdm(range(0, len(input_signal), partition_size), desc=f"Processing {os.path.basename(input_wav)}"):
        input_block = input_signal[i:i + partition_size]
        if len(input_block) < partition_size:
            input_block = np.pad(input_block, (0, partition_size - len(input_block)), mode='constant')
        
        # FFT 
        input_fft = np.fft.rfft(input_block)
        convolved = np.zeros_like(input_fft, dtype=complex)
        for j in range(num_partitions):
            convolved += input_fft * ir_partitions[j]
        
        # IFFT 
        output_ifft = np.fft.irfft(convolved, n=partition_size)
        
        start = i
        end = i + partition_size
        output_signal[start:end] += output_ifft[:partition_size]
    

    sf.write(output_wav, output_signal, input_sr)
    print(f"Saved: {output_wav}")

def batch_convolution(input_folder, ir_folder, output_folder, partition_size=1024):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    input_wavs = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    ir_wavs = [f for f in os.listdir(ir_folder) if f.endswith('.wav')]
    
    for input_wav in input_wavs:
        for ir_wav in ir_wavs:
            output_wav = os.path.join(
                output_folder,
                f"{os.path.splitext(input_wav)[0]}_{os.path.splitext(ir_wav)[0]}.wav"
            )
            partitioned_fft_convolution(
                os.path.join(input_folder, input_wav),
                os.path.join(ir_folder, ir_wav),
                output_wav,
                partition_size=partition_size
            )
            
            #ir_params = extract_ir_parameters(os.path.join(ir_folder, ir_wav))
            #print(f"Extracted IR Parameters for {ir_wav}: {ir_params}")




input_path = "data/original/1.wav"
ir_path = "IRs/Studio Nord Spring 3,0 flat.wav"
output_path = "data/synth/3.0_flat_convolved.wav"
ir_folder = "IRs"

def create_json_parameters(ir_folder):
    #partitioned_fft_convolution(input_path, ir_path, output_path)
    data = {}
    for ir_path in os.listdir(ir_folder):
        ir_params = extract_ir_parameters(os.path.join(ir_folder, ir_path))
        root, ext = os.path.splitext(ir_path)
        data[root] = ir_params

    with open(f"data/ir_parameters.json", "w+") as json_file:
        json.dump(data, json_file)


create_json_parameters(ir_folder)


batch_convolution("data/original","IRs","data/synth")