import os
import numpy as np
import soundfile as sf
import time
from pedalboard import Pedalboard, Convolution
from tqdm import tqdm

def pedalboard_convolution(input_wav, ir_wav, output_wav):
    # 1) Leer señales y comprobar SR
    x, sr_x = sf.read(input_wav)
    h, sr_h = sf.read(ir_wav)
    if sr_x != sr_h:
        raise ValueError(f"Sampling rates do not match: {sr_x} vs {sr_h}")

    # 2) Asegurar mono (primer canal si hay más)
    if x.ndim > 1:
        x = x[:, 0]
    if h.ndim > 1:
        h = h[:, 0]

    # 3) Crear pedalboard con plugin de Convolution
    board = Pedalboard([ Convolution(ir_wav) ])

    # 4) Procesar el audio y hacer flush para sacar la cola
    #    - Primera pasada: la señal real, reseteando el buffer interno
    y = board.process(x, sr_x, reset=True)
    #    - Segunda pasada: buffer de ceros para vaciar la cola (sin reset)
    zeros = np.zeros(len(h), dtype=x.dtype)
    tail  = board.process(zeros, sr_x, reset=False)

    # 5) Concatenar señal + cola y normalizar
    y_full = np.concatenate([y, tail])
    max_abs = np.max(np.abs(y_full))
    if max_abs > 1e-9 and max_abs > 1.0:
        y_full = y_full / max_abs

    # 6) Guardar resultado con cola incluida
    sf.write(output_wav, y_full, sr_x)
    print(f"Saved convolved file with tail (Pedalboard): {output_wav}")

# if __name__ == "__main__":
#     input_wav = "data/original/1.wav"
#     ir_wav    = "IRs/Studio Nord Spring 3,0 flat.wav"
#     os.makedirs("results_pedalboard", exist_ok=True)

#     # Procesar un archivo de ejemplo
#     pedalboard_convolution(
#         input_wav,
#         ir_wav,
#         os.path.join("results_pedalboard", "1_pedal_with_tail.wav")
#     )



def process_folder(input_dir: str, ir_wav: str, output_dir: str):
    """Procesa todos los WAV en input_dir con la IR dada y guarda en output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    for filename in tqdm(os.listdir(input_dir), desc="Procesando archivos WAV"):
        if not filename.lower().endswith('.wav'):
            continue
        input_path = os.path.join(input_dir, filename)
        base, _ = os.path.splitext(filename)
        output_name = f"{base}.wav"
        output_path = os.path.join(output_dir, output_name)
        pedalboard_convolution(input_path, ir_wav, output_path)
    elapsed = time.time() - start_time   # ← fin del cronómetro
    print(f"Pedalboard Process Time: {elapsed:.2f} seconds")

# Ex:
process_folder(
    input_dir="data/original",
    ir_wav="IRs/Studio Nord Spring 3,5 flat.wav",
    output_dir="data/pedal_time"
)
