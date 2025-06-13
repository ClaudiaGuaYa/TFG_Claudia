import os
import numpy as np
import soundfile as sf
import json
from tqdm import tqdm


def partitioned_fft_convolution(input_wav, ir_wav, output_wav, partition_size=1024):
    # Leemos señales y comprobamos la tasa de muestreo
    input_signal, input_sr = sf.read(input_wav)
    ir_signal, ir_sr = sf.read(ir_wav)
    if input_sr != ir_sr:
        raise ValueError(f"Sampling rates do not match: {input_sr} vs {ir_sr}")
    
    # Convertir IR a mono si es necesario
    if ir_signal.ndim > 1:
        ir_signal = ir_signal[:, 0]
    
    # Definir FFT size (L = 2·K) para overlap-save
    fft_size = 2 * partition_size
    
    # Particionar la IR: forzamos que la longitud sea múltiplo de partition_size
    remainder = len(ir_signal) % partition_size
    pad_width = partition_size - remainder if remainder != 0 else 0
    ir_padded = np.pad(ir_signal, (0, pad_width), mode='constant')
    num_partitions = len(ir_padded) // partition_size
    
    # Calculamos la FFT de cada partición (cero-padding a longitud fft_size)
    ir_partitions = np.array([
        np.fft.rfft(ir_padded[i * partition_size:(i + 1) * partition_size], n=fft_size)
        for i in range(num_partitions)
    ])
    
    # Para aplicar overlap-save, preparamos la señal de entrada:
    # Se añade un "overlap" inicial de partition_size ceros
    padded_input = np.pad(input_signal, (partition_size, 0), mode='constant')
    num_blocks = (len(padded_input) - partition_size) // partition_size
    # La salida total tendrá longitud: len(input_signal) + len(ir_signal) - 1
    output_length = len(input_signal) + len(ir_signal) - 1
    output_signal = np.zeros(output_length)
    
    # Procesamos cada bloque (ventana de longitud fft_size = 2·partition_size)
    for b in tqdm(range(num_blocks), desc=f"Processing {os.path.basename(input_wav)}"):
        start_index = b * partition_size
        block = padded_input[start_index : start_index + fft_size]
        if len(block) < fft_size:
            block = np.pad(block, (0, fft_size - len(block)), mode='constant')
            
        # FFT de la ventana
        block_fft = np.fft.rfft(block, n=fft_size)
        
        # Acumulamos la convolución en el dominio de la frecuencia
        conv_freq = np.zeros_like(block_fft, dtype=complex)
        for j in range(num_partitions):
            conv_freq += block_fft * ir_partitions[j]
        
        # Inversa de la FFT para obtener el bloque en tiempo
        block_time = np.fft.irfft(conv_freq, n=fft_size)
        # Extraer la parte válida: últimos fft_size - partition_size = partition_size muestras
        valid_output = block_time[partition_size:]
        # Ponderar dividiendo por el número de particiones (corrección de ganancia)
        valid_output = valid_output / num_partitions
        
        # Normalización basada en la energía: ajustamos para que la energía de la salida
        # coincida (aproximadamente) con la de la parte de entrada correspondiente
        input_block = block[partition_size:]
        energy_in = np.sum(input_block**2)
        energy_out = np.sum(valid_output**2)
        if energy_out > 0:
            norm_factor = np.sqrt(energy_in / energy_out)
        else:
            norm_factor = 1.0
        valid_output *= norm_factor
        
        # Ubicar el bloque válido en la señal de salida (con solapamiento en bloque)
        out_start = b * partition_size
        out_end = out_start + partition_size
        if out_end > len(output_signal):
            out_end = len(output_signal)
            valid_output = valid_output[:out_end - out_start]
        output_signal[out_start:out_end] += valid_output  # En overlap-add se suma
        
    # Opcional: si la señal excede el full scale (1), se normaliza globalmente
    max_val = np.max(np.abs(output_signal))
    if max_val > 1:
        output_signal = output_signal / max_val
    
    sf.write(output_wav, output_signal, input_sr)
    print(f"Saved: {output_wav}")
