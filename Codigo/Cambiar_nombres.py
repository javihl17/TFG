import multiprocessing
import shutil
import random
import os

save_path = "DATASETS/Dataset_3/Balanced/Fold3/"

def copiar(path, imagen, carpeta):
    #print(imagen)
    #print(save_path+"Train/"+carpeta+"/"+imagen)
    if random.randint(0, 100) <= 70:
        shutil.copy("DATASETS/Dataset_3/Balanced/Fold1/Train/"+carpeta+"/"+imagen, save_path+"Train/"+carpeta+"/"+imagen)
    else:
        shutil.copy("DATASETS/Dataset_3/Balanced/Fold1/Train/"+carpeta+"/"+imagen, save_path + "Test/"+carpeta+"/" + imagen)

if __name__ == '__main__':
    contenido = os.listdir("DATASETS/Dataset_3/Balanced/Fold1/Train")
    for carpeta in contenido:
        cuenta = 0
        imagenes = os.listdir("DATASETS/Dataset_3/Balanced/Fold1/Train/"+carpeta+"/")
        random.shuffle(imagenes)
        for i in imagenes:
            print(cuenta)
            copiar("DATASETS/Dataset_3/Balanced/Fold1/Train/"+carpeta+"/"+i, i, carpeta)
            cuenta = cuenta + 1
