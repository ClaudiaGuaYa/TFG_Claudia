#!/usr/bin/env python3
import os
import numpy as np
import soundfile as sf
from pedalboard import Pedalboard, Convolution
from scipy import signal
import matplotlib.pyplot as plt
import json
from tqdm import tqdm


def pedalboard_convolution(input_wav: str, ir_wav: str) -> np.ndarray:
    """Aplica convolución de reverb con Pedalboard y devuelve la señal con cola incluida."""
    # Leer señales y comprobar SR
    x, sr_x = sf.read(input_wav)
    h, sr_h = sf.read(ir_wav)
    if sr_x != sr_h:
        raise ValueError(f"Sampling rates do not match: {sr_x} vs {sr_h}")

    # Forzar mono (primer canal si hay más)
    if x.ndim > 1:
        x = x[:, 0]
    if h.ndim > 1:
        h = h[:, 0]

    # Configurar Pedalboard con plugin de Convolution
    board = Pedalboard([Convolution(ir_wav)])

    # Procesar audio (primera pasada) y resetear buffer
    y = board.process(x, sr_x, reset=True)
    # Pasada de ceros para extraer cola sin resetear
    zeros = np.zeros(len(h), dtype=x.dtype)
    tail = board.process(zeros, sr_x, reset=False)

    # Concatenar señal + cola
    y_full = np.concatenate([y, tail])
    return y_full, sr_x


def normalize_to_reference(ref_wav: str, target: np.ndarray, sr: int, method: str = 'peak') -> np.ndarray:
    """
    Normaliza 'target' para que coincida con el nivel de 'ref_wav'.
    method = 'peak' o 'rms'.
    """
    ref, sr_r = sf.read(ref_wav)
    if sr_r != sr:
        raise ValueError(f"Sampling rates do not match: {sr_r} vs {sr}")
    if ref.ndim > 1:
        ref = ref[:, 0]
    # Cálculo de niveles
    if method == 'peak':
        peak_ref = np.max(np.abs(ref))
        peak_tgt = np.max(np.abs(target))
        if peak_tgt > 0:
            target = target * (peak_ref / peak_tgt)
    elif method == 'rms':
        rms_ref = np.sqrt(np.mean(ref**2))
        rms_tgt = np.sqrt(np.mean(target**2))
        if rms_tgt > 0:
            target = target * (rms_ref / rms_tgt)
    else:
        raise ValueError("method must be 'peak' or 'rms',or 'both'")
    return target


def process_folder(input_dir: str, ir_wav: str, output_dir: str,
                   reference_dir: str = None, norm_method: str = 'peak', compute_spec: bool = False):
    """
    Procesa todos los WAV de input_dir aplicando convolución con ir_wav.
    Si reference_dir se facilita, normaliza cada output a su correspondiente WAV en reference_dir.
    Si compute_spec=True, genera espectrogramas de cada par (ref vs pedalboard) en output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    if compute_spec:
        spec_dir = os.path.join(output_dir, 'spectrograms')
        os.makedirs(spec_dir, exist_ok=True)

    for fn in tqdm(sorted(os.listdir(input_dir)), desc="Procesando señales"):
        if not fn.lower().endswith('.wav'):
            continue
        inp = os.path.join(input_dir, fn)
        # Convolución
        y_full, sr = pedalboard_convolution(inp, ir_wav)

        # Normalización si hay referencia
        if reference_dir:
            ref_path = os.path.join(reference_dir, fn)
            if os.path.exists(ref_path):
                y_full = normalize_to_reference(ref_path, y_full, sr, method=norm_method)

        # Guardar WAV resultante
        out_name = os.path.splitext(fn)[0] + f'_pedal_norm_{norm_method}.wav'
        out_path = os.path.join(output_dir, out_name)
        sf.write(out_path, y_full, sr)

        # Generar espectrograma si se solicita
        if compute_spec and reference_dir:
            # Espectrogramas de referencia y de pedalboard
            for label, wav_path in [('ref', os.path.join(reference_dir, fn)),
                                     ('pedal', out_path)]:
                y, _ = sf.read(wav_path)
                if y.ndim > 1:
                    y = y[:, 0]
                f, t, Sxx = signal.spectrogram(y, sr, nperseg=1024, noverlap=512)
                Sxx_db = 10 * np.log10(Sxx + 1e-10)
                plt.figure(figsize=(6,4))
                plt.pcolormesh(t, f, Sxx_db, shading='gouraud')
                plt.ylabel('Freq [Hz]')
                plt.xlabel('Time [s]')
                plt.title(f"{fn} - {label}")
                plt.colorbar(label='dB')
                spec_path = os.path.join(spec_dir, f"{os.path.splitext(fn)[0]}_{label}.png")
                plt.tight_layout()
                plt.savefig(spec_path)
                plt.close()

    print(f"Procesamiento completado. Resultados en: {output_dir}")



# process_folder(
#     input_dir="data/original",
#     ir_wav="IRs/Studio Nord Spring 3,0 flat.wav",
#     output_dir="results_pedalboard/pedal_norm_RMS",
#     reference_dir="data/synth_reverb_cola",
#     norm_method="rms",      
#     #compute_spec=True  
# )


# diferencia de nivel en dBs

def level_difference_db(ref: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Calcula diferencia de nivel (peak y RMS) en dB entre ref y target."""
    min_len = min(len(ref), len(target))
    ref, target = ref[:min_len], target[:min_len]
    # Peak diff
    peak_ref = np.max(np.abs(ref))
    peak_tgt = np.max(np.abs(target))
    peak_diff_db = 20 * np.log10(peak_tgt / peak_ref) if peak_ref > 0 else 0.0
    # RMS diff
    rms_ref = np.sqrt(np.mean(ref**2))
    rms_tgt = np.sqrt(np.mean(target**2))
    rms_diff_db = 20 * np.log10(rms_tgt / rms_ref) if rms_ref > 0 else 0.0
    return peak_diff_db, rms_diff_db


# def compute_level_differences(reference_dir: str, target_dir: str, output_json: str):
#     """Recorre todos los WAV de reference_dir vs target_dir y guarda un JSON con diferencias de nivel en dB."""
#     results = {}
#     for fn in sorted(os.listdir(reference_dir)):
#         if not fn.lower().endswith('.wav'):
#             continue
#         ref_path = os.path.join(reference_dir, fn)
#         tgt_path = os.path.join(target_dir, fn)
#         if not os.path.exists(tgt_path):
#             print(f"WARNING: no se encontró target para {fn}, se omite.")
#             continue
#         # Cargar señales
#         ref, sr = sf.read(ref_path)
#         tgt, sr2 = sf.read(tgt_path)
#         if sr2 != sr:
#             raise ValueError(f"Sampling rates do not match for {fn}: {sr} vs {sr2}")
#         # Mono
#         if ref.ndim > 1:
#             ref = ref[:, 0]
#         if tgt.ndim > 1:
#             tgt = tgt[:, 0]
#         # Calcular diferencias
#         peak_db, rms_db = level_difference_db(ref, tgt)
#         results[fn] = {
#             'peak_diff_db': peak_db,
#             'rms_diff_db': rms_db
#         }
#     # Guardar JSON
#     with open(output_json, 'w') as f:
#         json.dump(results, f, indent=2)
#     print(f"Guardado JSON de diferencias de nivel: {output_json}")


#NO HAY CASI DIFERENCIA ENTRE REFERENCIA FARINA Y LOS YA NORMALIZADOS AL MISMO NIVEL, NO SENSE
# def compute_level_differences(reference_dir: str, target_root: str, output_json: str,
#                                peak_subfolder: str = 'pedal_norm_peak',
#                                rms_subfolder: str = 'pedal_norm_RMS'):
#     """
#     Para cada WAV en reference_dir (e.g. '1.wav'), busca los archivos normalizados en las subcarpetas
#     peak_subfolder y rms_subfolder dentro de target_root, calcula las diferencias de nivel en dB
#     (peak y RMS) y guarda todo en un JSON único.

#     El JSON tendrá la forma:
#     {
#       "1": {
#          "peak_norm": {"peak_diff_db": x.xx, "rms_diff_db": y.yy},
#          "rms_norm": {"peak_diff_db": a.aa, "rms_diff_db": b.bb}
#       },
#       ...
#     }
#     """
#     results = {}
#     peak_dir = os.path.join(target_root, peak_subfolder)
#     rms_dir  = os.path.join(target_root, rms_subfolder)

#     for fn in sorted(os.listdir(reference_dir)):
#         if not fn.lower().endswith('.wav'):
#             continue
#         base = os.path.splitext(fn)[0]  # e.g. '1'
#         # Paths de archivos normalizados
#         peak_file = os.path.join(peak_dir, f"{base}_pedal_norm_peak.wav")
#         rms_file  = os.path.join(rms_dir,  f"{base}_pedal_norm_rms.wav")
#         entry = {}
#         # Cargar referencia
#         ref_path = os.path.join(reference_dir, fn)
#         ref, sr = sf.read(ref_path)
#         if ref.ndim > 1:
#             ref = ref[:,0]

#         # Peak norm
#         if os.path.exists(peak_file):
#             tgt_peak, sr2 = sf.read(peak_file)
#             if tgt_peak.ndim > 1:
#                 tgt_peak = tgt_peak[:,0]
#             if sr2 != sr:
#                 raise ValueError(f"Sampling rate mismatch {peak_file}: {sr2} vs {sr}")
#             pd_peak, rd_peak = level_difference_db(ref, tgt_peak)
#             entry['peak_norm'] = {'peak_diff_db': round(pd_peak,2), 'rms_diff_db': round(rd_peak,2)}
#         else:
#             print(f"No encontrado peak_norm para {base} en {peak_dir}")

#         # RMS norm
#         if os.path.exists(rms_file):
#             tgt_rms, sr3 = sf.read(rms_file)
#             if tgt_rms.ndim > 1:
#                 tgt_rms = tgt_rms[:,0]
#             if sr3 != sr:
#                 raise ValueError(f"Sampling rate mismatch {rms_file}: {sr3} vs {sr}")
#             pd_rms, rd_rms = level_difference_db(ref, tgt_rms)
#             entry['rms_norm'] = {'peak_diff_db': round(pd_rms,2), 'rms_diff_db': round(rd_rms,2)}
#         else:
#             print(f"No encontrado rms_norm para {base} en {rms_dir}")

#         if entry:
#             results[base] = entry

#     # Guardar JSON
#     with open(output_json, 'w') as f:
#         json.dump(results, f, indent=2)
#     print(f"JSON de diferencias guardado en: {output_json}")


# # Ejemplo de uso directo en script
# # compute_level_differences(
# #     reference_dir='data/synth_reverb_cola',
# #     target_dir='results_pedalboard',
# #     output_json='results_pedalboard/level_differences.json'
# # )

# compute_level_differences(
#     reference_dir="data/synth_reverb_cola",
#     target_root="results_pedalboard/pedal_no_norm",
#     output_json="results_pedalboard/level_differences.json"
# )



def compute_level_differences(reference_dir: str, target_dir: str, output_json: str):
    """
    Para cada WAV en reference_dir (e.g. '1.wav'), compara con el mismo nombre en target_dir
    (archivos no normalizados) y calcula la diferencia de nivel en dB (peak y RMS),
    guardando los resultados en un JSON.

    JSON resultante:
    {
      "1": {"peak_diff_db": x.xx, "rms_diff_db": y.yy},
      "2": {…}
    }
    """
    results = {}
    for fn in sorted(os.listdir(reference_dir)):
        if not fn.lower().endswith('.wav'):
            continue
        base = os.path.splitext(fn)[0]
        ref_path = os.path.join(reference_dir, fn)
        tgt_path = os.path.join(target_dir, fn)
        if not os.path.exists(tgt_path):
            print(f"No encontrado target para {fn} en {target_dir}")
            continue
        # Cargar señal de referencia
        ref, sr = sf.read(ref_path)
        if ref.ndim > 1:
            ref = ref[:, 0]
        # Cargar señal objetivo
        tgt, sr2 = sf.read(tgt_path)
        if tgt.ndim > 1:
            tgt = tgt[:, 0]
        if sr2 != sr:
            raise ValueError(f"Sampling rate mismatch para '{fn}': {sr2} vs {sr}")
        # Calcular diferencias
        peak_db, rms_db = level_difference_db(ref, tgt)
        results[base] = {
            'peak_diff_db': round(peak_db, 2),
            'rms_diff_db': round(rms_db, 2)
        }
    # Guardar JSON
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"JSON de diferencias guardado en: {output_json}")


compute_level_differences(
    reference_dir="data/synth_reverb_cola",
    target_dir="results_pedalboard/pedal_no_norm",
    output_json="results_pedalboard/level_differences.json"
)
