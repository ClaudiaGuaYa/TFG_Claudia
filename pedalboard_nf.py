import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
from pedalboard import Pedalboard, Convolution

def extract_ir_parameters(ir_wav):
    """
    Extrae parámetros de un archivo WAV de respuesta al impulso (IR), incluyendo el tiempo de decaimiento y características espectrales.
    :param ir_wav: Ruta al archivo WAV de respuesta al impulso
    :return: Diccionario con los parámetros extraídos
    """
    ir_signal, ir_sr = sf.read(ir_wav)
    ir_signal = ir_signal / np.max(np.abs(ir_signal))  # Normalizar IR
    
    # Calcular el tiempo de decaimiento (estimación aproximada de RT60)
    energy = np.cumsum(ir_signal[::-1]**2)[::-1]
    decay_time = np.argmax(energy < max(energy) * 0.001) / ir_sr
    
    return {
        "sampling_rate": ir_sr,
        "decay_time": decay_time
    }

def pedalboard_partitioned_convolution(input_wav, ir_wav, output_wav):
    """
    Realiza la convolución FFT particionada usando Pedalboard.
    :param input_wav: Ruta al archivo WAV de entrada
    :param ir_wav: Ruta al archivo WAV de respuesta al impulso (IR)
    :param output_wav: Ruta para guardar el archivo WAV convolucionado
    """
    # Cargar audio de entrada y respuesta al impulso
    input_signal, input_sr = sf.read(input_wav)
    ir_signal, ir_sr = sf.read(ir_wav)
    
    # Si la IR es estéreo, convertirla a mono (se usa el primer canal)
    if ir_signal.ndim > 1:
        ir_signal = ir_signal[:, 0]
    
    # Convertir IR a float32 para compatibilidad con Convolution
    ir_signal = ir_signal.astype(np.float32)
    
    # Normalizar IR
    ir_signal = ir_signal / np.max(np.abs(ir_signal))
    ir_signal = ir_signal / np.sqrt(len(ir_signal))
    
    # Verificar que las tasas de muestreo coincidan
    if input_sr != ir_sr:
        raise ValueError(f"Las tasas de muestreo no coinciden: {input_sr} vs {ir_sr}")
    
    # Usar Pedalboard para la convolución
    board = Pedalboard([
        Convolution(ir_signal, sample_rate=ir_sr)
    ])
    
    output_signal = board(input_signal, input_sr)
    
    # Normalizar salida
    output_signal = output_signal / np.max(np.abs(output_signal))
    
    # Guardar salida
    sf.write(output_wav, output_signal, input_sr)
    print(f"Guardado: {output_wav}")

def batch_convolution(input_folder, ir_folder, output_folder):
    """
    Aplica la convolución de Pedalboard a todos los archivos WAV de entrada con cada IR.
    :param input_folder: Carpeta que contiene archivos WAV de entrada
    :param ir_folder: Carpeta que contiene archivos WAV de respuesta al impulso (IR)
    :param output_folder: Carpeta para guardar los archivos convolucionados
    """
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
            pedalboard_partitioned_convolution(
                os.path.join(input_folder, input_wav),
                os.path.join(ir_folder, ir_wav),
                output_wav
            )
            
            # Extraer parámetros de la IR
            ir_params = extract_ir_parameters(os.path.join(ir_folder, ir_wav))
            print(f"Parámetros extraídos de {ir_wav}: {ir_params}")

# Ejemplo de uso:
# Para procesar en lote:
# batch_convolution('ruta_a_la_carpeta_de_wav', 'ruta_a_la_carpeta_de_ir', 'ruta_a_la_carpeta_de_salida')


# Llamada individual (asegúrate de que las rutas sean correctas)
pedalboard_partitioned_convolution("data/original/1.wav", "IRs/Studio Nord Spring 1,5 flat.wav", "data/synth")
