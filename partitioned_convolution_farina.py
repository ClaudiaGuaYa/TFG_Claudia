
import os
import numpy as np
import soundfile as sf
from tqdm import tqdm


def partitioned_fft_convolution(input_wav, ir_wav, output_wav, partition_size=1024):

    x, sr_x = sf.read(input_wav)
    h, sr_h = sf.read(ir_wav)
    if sr_x != sr_h:
        raise ValueError(f"Sampling rates do not match: {sr_x} vs {sr_h}")
    
    # mono
    if x.ndim > 1:
        x = x[:, 0]
    if h.ndim > 1:
        h = h[:, 0]
    
    # get size
    L = partition_size
    fft_size = 2 * L  # overlap-save 2*L window
    len_x = len(x)
    len_h = len(h)
    
    # IR L partition
    remainder = len_h % L
    if remainder != 0:
        h = np.pad(h, (0, L - remainder), mode='constant')
    num_partitions = len(h) // L
    
    # FFT each partition
    H_partitions = []
    for i in range(num_partitions):
        h_i = h[i*L : (i+1)*L]
        H_partitions.append(np.fft.rfft(h_i, n=fft_size))
    H_partitions = np.array(H_partitions)

    padded_x = np.pad(x, (L, 0), mode='constant')
    out_len = len_x + len_h - 1
    y = np.zeros(out_len, dtype=np.float64)
    
    # blocks processed
    num_blocks = (len(padded_x) - L) // L  # overlapping L to L
    
    for b in tqdm(range(num_blocks), desc="Partitioned Convolution"):
        start_b = b * L
        block_time = padded_x[start_b : start_b + fft_size]
        if len(block_time) < fft_size:
            block_time = np.pad(block_time, (0, fft_size - len(block_time)), mode='constant')
        
        # FFT block
        X_b = np.fft.rfft(block_time, n=fft_size)
        
        # each IR partition
        for j in range(num_partitions):
            Y_freq = X_b * H_partitions[j]
            y_time = np.fft.irfft(Y_freq, n=fft_size)

            valid_time = y_time[L:]
            
            out_start = (b + j) * L
            out_end   = out_start + L
            
            # avoid exceeding the output size
            if out_start < out_len:
                if out_end > out_len:
                    valid_time = valid_time[: out_len - out_start]
                    out_end = out_len
                y[out_start : out_end] += valid_time
    
    # normalization (avoids peaks > 1)
    max_abs = np.max(np.abs(y))
    if max_abs > 1e-9:  
        if max_abs > 1.0:
            y /= max_abs
    
    sf.write(output_wav, y, sr_x)
    print(f"Saved convolved file: {output_wav}")

