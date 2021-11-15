import librosa as lb
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt

# Definir el path donde se encuentran los .wav de TES
TES_path = 'C:/Users/javih/Desktop/TFG/dataverse_files'
os.chdir(TES_path)

# Listado de emociones en el dataset TES
TES_emotions = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]

# @input: Tamaño en muestras deseado de los audios
muestras = 20000

datos = []
etiquetas = []


# Recorrer los .wav del directorio TES
for file in os.listdir():
    # Buscar coincidencias con las emociones para clasificarlos
    for emotion in TES_emotions:
        if file.find(emotion) >= 0:
            y, sr = lb.load(TES_path + '/' + file)
            y, _ = lb.effects.trim(y)
            duration = len(y) * (1/sr)

            while len(y) >= muestras:
                particion = y[0:muestras]
                datos.append(np.append(particion, sr))
                etiquetas.append(TES_emotions.index(emotion))
                y = y[muestras:len(y)]

# Definir el path donde se encuentran los .wav de Berlin
Berlin_path = 'C:/Users/javih/Desktop/TFG/download/wav'
os.chdir(Berlin_path)

# Listado de emociones en el dataset Berlin
Berlin_emotions = ["anger", "boredom", "disgust", "anxiety/fear", "hapiness", "sadness", "neutral"]

# Recorrer los .wav del directorio Berlin
for file in os.listdir():
    # Buscar coincidencias con las emociones para clasificarlos
    y, sr = lb.load(Berlin_path + '/' + file)
    duration = len(y) * (1/sr)

    while len(y) >= muestras:
        particion = y[0:muestras]
        datos.append(np.append(particion, sr))
        if file.find('W') > 0:
            etiquetas.append(0)
        elif file.find('L') > 0:
            etiquetas.append(7)
        elif file.find('E') > 0:
            etiquetas.append(1)
        elif file.find('A') > 0:
            etiquetas.append(2)
        elif file.find('F') > 0:
            etiquetas.append(3)
        elif file.find('T') > 0:
            etiquetas.append(6)
        elif file.find('N') > 0:
            etiquetas.append(4)
        y = y[muestras:len(y)]

# Definir el path donde se encuentran los .wav de Radvess
Radvess_path = 'C://Users//javih//Desktop//TFG//Audio_Speech_Actors_01-24'
os.chdir(Radvess_path)

# Listado de emociones en el dataset Berlin
Radvess_emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

#Lista de los subdirectorios
subdis = os.listdir()

#Loop que recorre los subdirectorios
for subd in subdis:
    os.chdir(Radvess_path+"//"+subd)
    print(Radvess_path+"//"+subd)
    for file in os.listdir():
        y, sr = lb.load(Radvess_path + '//' + subd + '/' + file)
        duration = len(y) * (1 / sr)

        while len(y) >= muestras:
            particion = y[0:muestras]
            datos.append(np.append(particion, sr))
            codigo = int (file[7:8])

            if codigo == 1:
                etiquetas.append(4)
            elif codigo == 2:
                etiquetas.append(8)
            elif codigo == 3:
                etiquetas.append(3)
            elif codigo == 4:
                etiquetas.append(6)
            elif codigo == 5:
                etiquetas.append(0)
            elif codigo == 6:
                etiquetas.append(2)
            elif codigo == 7:
                etiquetas.append(1)
            elif codigo == 8:
                etiquetas.append(5)

            y = y[muestras:len(y)]

save_path = 'C://Users//javih//Desktop//TFG/'
os.chdir(save_path)

import pickle
archivo_datos = open('trimmed_datos.dat', 'wb')
pickle.dump(datos, archivo_datos)
archivo_datos.close()

archivo_etiquetas = open('trimmed_etiquetas.dat', 'wb')
pickle.dump(etiquetas, archivo_etiquetas)
archivo_etiquetas.close()
