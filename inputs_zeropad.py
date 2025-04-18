import os
import numpy as np
from scipy.io import wavfile

def zero_pad_inputs(
    input_folder: str,
    convolved_folder: str,
    output_folder: str,
    epsilon: float = 1e-16
):

    os.makedirs(output_folder, exist_ok=True)
    
    input_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.wav')]
    conv_files = set(f for f in os.listdir(convolved_folder) if f.lower().endswith('.wav'))

    for in_file in input_files:
        if in_file in conv_files:
            in_path = os.path.join(input_folder, in_file)
            conv_path = os.path.join(convolved_folder, in_file)

            sr_in, data_in = wavfile.read(in_path)
            sr_conv, data_conv = wavfile.read(conv_path)

            if sr_in != sr_conv:
                print(f"[Warning] {in_file}: SR input={sr_in}, SR convolved={sr_conv}. "
                      "They should match to avoid problems.")

            len_in = data_in.shape[0]
            len_conv = data_conv.shape[0]

            if len_conv > len_in:
                diff = len_conv - len_in
                if np.issubdtype(data_in.dtype, np.floating):
                    # If float (float32, float64,...), epsilon is used
                    pad_values = np.full(diff, epsilon, dtype=data_in.dtype)
                else:
                    # If int16, int32,..., 0s are written
                    pad_values = np.zeros(diff, dtype=data_in.dtype)
                
                data_in_padded = np.concatenate((data_in, pad_values))
            else:
                # no padding needed
                data_in_padded = data_in

            out_path = os.path.join(output_folder, in_file)
            wavfile.write(out_path, sr_in, data_in_padded)
            print(f"→ Saved with padding: {out_path}")
        else:
            print(f"No convolved found for {in_file}; skipped.")