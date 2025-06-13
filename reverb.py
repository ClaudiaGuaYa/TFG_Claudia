import os
import numpy as np
import soundfile as sf
from tqdm import tqdm


def partitioned_fft_convolution(input_wav, ir_wav, output_wav, partition_size=1024):
    # 1) Leer señales y comprobar SR
    x, sr_x = sf.read(input_wav)
    h, sr_h = sf.read(ir_wav)
    if sr_x != sr_h:
        raise ValueError(f"Sampling rates do not match: {sr_x} vs {sr_h}")
    
    # 2) Asegurar mono (ejemplo: tomar el primer canal)
    if x.ndim > 1:
        x = x[:, 0]
    if h.ndim > 1:
        h = h[:, 0]
    
    # 3) Determinar tamaños
    L = partition_size
    fft_size = 2 * L  # se emplea overlap-save con ventana de 2*L
    len_x = len(x)
    len_h = len(h)
    
    # 4) Particionar la IR a longitud L (padding si no es múltiplo exacto)
    remainder = len_h % L
    if remainder != 0:
        h = np.pad(h, (0, L - remainder), mode='constant')
    num_partitions = len(h) // L
    
    # 5) FFT de cada partición (longitud fft_size)
    H_partitions = []
    for i in range(num_partitions):
        h_i = h[i*L : (i+1)*L]
        H_partitions.append(np.fft.rfft(h_i, n=fft_size))
    H_partitions = np.array(H_partitions)
    
    # 6) Padding de la entrada (L ceros al inicio)
    #    Esto permite hacer overlap-save cómodamente
    padded_x = np.pad(x, (L, 0), mode='constant')
    
    # 7) Tamaño de la salida (longitud = len_x + len_h - 1)
    out_len = len_x + len_h - 1
    y = np.zeros(out_len, dtype=np.float64)
    
    # 8) Número total de “bloques” que procesaremos
    #    Cada bloque real de audio son L muestras de la entrada
    #    pero en la práctica se extraen 2*L (overlap) para la FFT
    num_blocks = (len(padded_x) - L) // L  # solapando de L en L
    
    # 9) Bucle principal
    for b in tqdm(range(num_blocks), desc="Partitioned Convolution"):
        # Extraer bloque de 2*L de la entrada
        start_b = b * L
        block_time = padded_x[start_b : start_b + fft_size]
        if len(block_time) < fft_size:
            block_time = np.pad(block_time, (0, fft_size - len(block_time)), mode='constant')
        
        # FFT del bloque (rfft => array complejo)
        X_b = np.fft.rfft(block_time, n=fft_size)
        
        # Para cada partición de la IR…
        for j in range(num_partitions):
            # 1) Multiplicación en frecuencia
            Y_freq = X_b * H_partitions[j]
            
            # 2) IFFT para obtener el bloque temporal
            y_time = np.fft.irfft(Y_freq, n=fft_size)
            
            # 3) En overlap-save, la “parte útil” son las últimas L muestras
            #    (descartamos las primeras L como overlap)
            valid_time = y_time[L:]
            
            # 4) Colocar este resultado con el offset (b + j) * L
            out_start = (b + j) * L
            out_end   = out_start + L
            
            # Evitar pasarse del tamaño de la salida
            if out_start < out_len:
                if out_end > out_len:
                    valid_time = valid_time[: out_len - out_start]
                    out_end = out_len
                # Acumular
                y[out_start : out_end] += valid_time
    
    # 10) Normalización global (evitar picos > 1)
    max_abs = np.max(np.abs(y))
    if max_abs > 1e-9:  # evitar división por cero si la señal es minúscula
        if max_abs > 1.0:
            y /= max_abs
    
    # 11) Guardar resultado
    sf.write(output_wav, y, sr_x)
    print(f"Saved convolved file: {output_wav}")



input_path = "data/original/1.wav"
ir_path = "IRs/Studio Nord Spring 3,0 flat.wav"
output_path = "data/1.5_flat_convolved1.wav"
ir_folder = "IRs"

#partitioned_fft_convolution(input_path, ir_path, output_path)









#same length outputs 18''
def single_convolution(input_folder, ir_path, output_folder, partition_size=1024):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    input_wavs = [f for f in os.listdir(input_folder) if f.endswith('.wav')]

    for input_wav in input_wavs:
        in_path = os.path.join(input_folder, input_wav)
        out_path = os.path.join(output_folder, os.path.splitext(input_wav)[0] + ".wav")

        # 1) Leemos la longitud de la señal original
        x, sr = sf.read(in_path)
        original_length = len(x)

        # 2) Aplicamos la convolución
        partitioned_fft_convolution(
            in_path,
            ir_path,
            out_path,
            partition_size=partition_size
        )

        # 3) Releemos la salida generada
        y, sr_y = sf.read(out_path)

        # 4) Recortar o rellenar la salida para que coincida con la longitud original
        if len(y) > original_length:
            y = y[:original_length]
        elif len(y) < original_length:
            # Si por algún motivo fuera más corta, se puede rellenar con ceros
            padding = original_length - len(y)
            y = np.pad(y, (0, padding), mode='constant')

        # 5) Guardamos la versión recortada/sin cola extra
        sf.write(out_path, y, sr)

        print(f"Saved trimmed output: {out_path}")

single_convolution("data/original",ir_path,"data/synth_reverb_same_length")