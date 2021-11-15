import random
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
save_path_3 = 'C://Users//javih//Desktop//TFG//DATASETS//Dataset_3'
save_path_5 = 'C://Users//javih//Desktop//TFG//DATASETS//Dataset_5'

# Etiquetas
string_etiquetas_3 = ['Anger',  'Disgust', 'Fear',  'Happyness', 'Neutral', 'PS', 'Sadness']
string_etiquetas_5 = ['Anger',  'Disgust', 'Fear',  'Happyness', 'Neutral', 'PS', 'Sadness', 'Calm']


# Abrir los np arrays que contienen los .wav y las etiquetas
archivo_datos_3 = open('C://Users//javih//Desktop//TFG/trimmed_datos_RT.dat', 'rb')
datos_3 = pickle.load(archivo_datos_3)
a_datos_3 = np.asarray(datos_3)
archivo_etiquetas_3 = open('C://Users//javih//Desktop//TFG/trimmed_etiquetas_RT.dat', 'rb')
etiquetas_3 = pickle.load(archivo_etiquetas_3)

archivo_datos_5 = open('C://Users//javih//Desktop//TFG/trimmed_datos_RB.dat', 'rb')
datos_5 = pickle.load(archivo_datos_5)
a_datos_5 = np.asarray(datos_5)
archivo_etiquetas_5 = open('C://Users//javih//Desktop//TFG/trimmed_etiquetas_RB.dat', 'rb')
etiquetas_5 = pickle.load(archivo_etiquetas_5)


n_fft = 512 #tamaño de las ventanas que se pasan para aplicar la transformada simple de fourier
hop_length = 256 #cuanto es el salto de esas ventanas
n_mels = 128 # Numero de mels que se van a usar en el eje y. De esto va a depender el tamaño del eje y pero
            # no 1 a 1 ya que los mels correspondientes a frecuencias mayores abarcan más frecuencias.


def procesar_wav_3(y, sr, etiqueta, ii):
    S = lb.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = lb.power_to_db(S, ref=np.max)  # Pasar a decibelios
    lb.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.axis('off')
    print(etiqueta)
    if random.randint(0, 100) <= 70:
        plt.savefig(save_path_3+'//Train//'+ string_etiquetas_3[etiqueta] + '/' + string_etiquetas_3[etiqueta] + str(ii) + '.png', bbox_inches='tight')
    else:
        plt.savefig(save_path_3+'//Test//'+ string_etiquetas_3[etiqueta] + '/' + string_etiquetas_3[etiqueta] + str(ii) + '.png', bbox_inches='tight')

def procesar_wav_5(y, sr, etiqueta, ii):
    S = lb.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = lb.power_to_db(S, ref=np.max)  # Pasar a decibelios
    lb.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.axis('off')
    print(ii)
    if random.randint(0, 100) <= 70:
        plt.savefig(save_path_5+'//Train//'+ string_etiquetas_5[etiqueta] + '/' + string_etiquetas_5[etiqueta] + str(ii) + '.png', bbox_inches='tight')
    else:
        plt.savefig(save_path_5+'//Test//'+ string_etiquetas_5[etiqueta] + '/' + string_etiquetas_5[etiqueta] + str(ii) + '.png', bbox_inches='tight')


if __name__ == '__main__':
    ii = np.shape(a_datos_3)[0]
    n_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=n_cores)(delayed(procesar_wav_3)(entrada[:len(entrada)-1], entrada[len(entrada)-1], etiqueta,ii) for entrada,etiqueta,ii in zip(a_datos_3, etiquetas_3,range(5044, ii)))

    '''ii = np.shape(a_datos_5)[0]
    n_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=n_cores)(delayed(procesar_wav_5)(entrada[:len(entrada) - 1], entrada[len(entrada) - 1], etiqueta, ii) for entrada, etiqueta, ii in zip(a_datos_5, etiquetas_5, range(ii)))
    '''
'''for entrada,etiqueta in zip(a_datos, etiquetas):
    y = entrada[:len(entrada)-1]
    sr = entrada[len(entrada)-1]
    procesar_wav(y, sr, etiqueta)'''

archivo_datos_3.close()
archivo_datos_5.close()