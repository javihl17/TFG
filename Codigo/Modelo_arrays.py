from comet_ml import Experiment
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator as IDG, ImageDataGenerator
from tensorflow.keras import datasets, layers, models
import pandas
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

experiment = Experiment(
    api_key="AStTN6xS6FtPY0kMEeVFUvptp",
    project_name="cnn-tfg",
    workspace="javihl17",
)

epochs = 60
l_rate = 5e-5

def create_dataframe():
    # Directorio de donde se obtienen los np arrays
    open_path = 'C://Users//javih//Desktop//TFG//DATASETS//Dataset_4/'
    archivo_datos = open(open_path + 'datos_dataset_4.dat', 'rb')
    datos = pickle.load(archivo_datos)
    a_datos = np.asarray(datos)
    print(np.shape(a_datos))
    archivo_etiquetas = open(open_path + 'etiquetas_dataset_4.dat', 'rb')
    etiquetas = np.asarray(pickle.load(archivo_etiquetas))
    print(etiquetas)
    X_train, X_test, y_train, y_test = train_test_split(a_datos, etiquetas, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

def create_cnn():
    model = models.Sequential()

    # Normalizar la entrada
    model.add(layers.BatchNormalization(input_shape=(20, 79, 1)))

    # Filtro de convolucion
    '''model.add(layers.Conv2D(1, (5, 5), activation='relu', padding='same'))
    model.add(layers.Dropout(0.25))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Conv2D(2, (4, 4), activation='relu', padding='same'))
    model.add(layers.Dropout(0.25))'''

    # ----Capa densa
    model.add(layers.Flatten())
    model.add(layers.Dense(3, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(7, activation='softmax'))
    return model


modelo_cnn = create_cnn()
modelo_cnn.summary()
modelo_cnn.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'])

X_train, X_test, Y_train, Y_test = create_dataframe()
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)
historico = modelo_cnn.fit(x = X_train,y = Y_train, epochs=epochs, validation_data=(X_test, Y_test))