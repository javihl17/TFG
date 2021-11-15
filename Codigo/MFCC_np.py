import librosa as lb
import librosa.display
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import multiprocessing
from joblib import delayed, Parallel
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Directorio de donde se obtienen los np arrays
open_path = 'C://Users//javih//Desktop//TFG/'
os.chdir(open_path)

# Directorio en donde se encuentran las carpetas de las imágenes
save_path = 'C://Users//javih//Desktop//TFG//Imagenes/'

# Etiquetas
string_etiquetas = ['Anger',  'Disgust', 'Fear',  'Happyness', 'Neutral', 'PS', 'Sadness', 'Boredom', 'Calm']
cuenta = [0, 0, 0, 0, 0, 0, 0, 0, 0]

# Np para guardar los datos
datos_np = []
etiquetas_return = []

# Abrir los np arrays que contienen los .wav y las etiquetas
archivo_datos = open('trimmed_datos_RB.dat', 'rb')
datos = pickle.load(archivo_datos)
a_datos = np.asarray(datos)
archivo_etiquetas = open('trimmed_etiquetas_RB.dat', 'rb')
etiquetas = pickle.load(archivo_etiquetas)


n_fft = 512 #tamaño de las ventanas que se pasan para aplicar la transformada simple de fourier
hop_length = 256 #cuanto es el salto de esas ventanas
n_mels = 128 # Numero de mels que se van a usar en el eje y. De esto va a depender el tamaño del eje y pero
            # no 1 a 1 ya que los mels correspondientes a frecuencias mayores abarcan más frecuencias.


def procesar_wav(y, sr):
    S = lb.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = lb.power_to_db(S, ref=np.max)  # Pasar a decibelios
    return S_DB

ii = 0

if __name__ == '__main__':
    print(len(etiquetas))
    print(len(a_datos))
    for entrada in a_datos:
        print(ii)
        if cuenta[etiquetas[ii]] < 630:
            datos_np.append(procesar_wav(entrada[:len(entrada)-1], entrada[len(entrada)-1]))
            if etiquetas[ii] == 7:
                etiquetas_return.append(4)
            elif etiquetas[ii] == 8:
                etiquetas_return.append(7)
            else:
                etiquetas_return.append(etiquetas[ii])
        cuenta[etiquetas[ii]] = cuenta[etiquetas[ii]] + 1
        ii = ii + 1

print(cuenta)
print(len(datos_np))
print(len(etiquetas_return))

# Guardar los np en un .dat
archivo_datos_np = open('datos_dataset_6.dat', 'wb')
archivo_etiquetas_np = open('etiquetas_dataset_6.dat', 'wb')
pickle.dump(datos_np, archivo_datos_np)
pickle.dump(etiquetas_return, archivo_etiquetas_np)

archivo_datos.close()
archivo_etiquetas_np.close()
archivo_datos_np.close()
