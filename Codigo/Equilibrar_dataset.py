import multiprocessing
import shutil
import random
import os

save_path = "DATASETS//Dataset_5"
conjunto = "Train"
cantidad = 440

def copiar(path, imagen, carpeta):
    #print(imagen)
    #print(save_path+"Train/"+carpeta+"/"+imagen)
    shutil.copy(save_path+"//"+conjunto+"//"+carpeta+"/"+imagen, save_path+"//Balanced//"+"//"+conjunto+"//"+carpeta+"/"+imagen)

if __name__ == '__main__':
    contenido = os.listdir(save_path+"//"+conjunto)
    for carpeta in contenido:
        cuenta = 0
        imagenes = os.listdir(save_path+"//"+conjunto+"//"+carpeta+"/")
        random.shuffle(imagenes)
        for i in imagenes:
            if cuenta <= cantidad:
                print(cuenta)
                copiar(save_path+"//"+conjunto+"//"+carpeta+"/"+i, i, carpeta)
                cuenta = cuenta + 1
            else:
                break