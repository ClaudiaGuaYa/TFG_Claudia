import os
import wave

def min_max_lenght_files(ruta_carpeta):
  
    data = []
    for archivo in os.listdir(ruta_carpeta):
        if archivo.lower().endswith('.wav'):
            ruta_archivo = os.path.join(ruta_carpeta, archivo)

            with wave.open(ruta_archivo, 'rb') as w:
                frames = w.getnframes()
                rate = w.getframerate()
                duracion = frames / float(rate)
                data.append((archivo, duracion))

    if not data:
        return (None, []), (None, [])
    
    min_duracion = min(data, key=lambda x: x[1])[1]
    max_duracion = max(data, key=lambda x: x[1])[1]
    
    archivos_min = [archivo for archivo, dur in data if dur == min_duracion]
    archivos_max = [archivo for archivo, dur in data if dur == max_duracion]
    
    return (min_duracion, archivos_min), (max_duracion, archivos_max)




carpeta = "data/input_zeropadding"
(dur_min, wavs_min), (dur_max, wavs_max) = min_max_lenght_files(carpeta)
print(f"Duración mínima: {dur_min} s -> Archivos: {wavs_min}")
print(f"Duración máxima: {dur_max} s -> Archivos: {wavs_max}")
