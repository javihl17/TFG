from comet_ml import Experiment
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator as IDG, ImageDataGenerator
from tensorflow.keras import datasets, layers, models
import pandas

experiment = Experiment(
    api_key="AStTN6xS6FtPY0kMEeVFUvptp",
    project_name="cnn-tfg",
    workspace="javihl17",
)

epochs = 150
l_rate = 1e-5


def create_dataframe():
    train_generator = ImageDataGenerator()
    train = train_generator.flow_from_directory(
        directory='Imagenes_trim_balanceadas/Train',
        target_size=(515, 389),
        color_mode="rgb",
        classes=None,
        class_mode="sparse",
        batch_size=24,
        shuffle=True,
        follow_links=False,
        subset='training')

    test_generator = ImageDataGenerator()
    test = test_generator.flow_from_directory(
        directory='Imagenes_trim_balanceadas/Test',
        target_size=(515, 389),
        color_mode="rgb",
        classes=None,
        batch_size=24,
        class_mode="sparse",
        shuffle=True,
        follow_links=False,
        subset='training')
    return train, test


def create_cnn():
    model = models.Sequential()

    # Normalizar la entrada
    model.add(layers.BatchNormalization(input_shape=(515, 389, 3)))

    # Filtro de convolucion
    model.add(layers.Conv2D(10, (5, 5), activation='relu', padding='same'))
    model.add(layers.Dropout(0.25))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(2, (4, 4), activation='relu', padding='same'))
    model.add(layers.Dropout(0.25))
    model.add(layers.MaxPooling2D((2, 2)))

    # ----Capa densa
    model.add(layers.Flatten())
    model.add(layers.Dense(21, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(7, activation='softmax'))
    return model


modelo_cnn = create_cnn()
modelo_cnn.summary()
modelo_cnn.compile(
    optimizer=tf.keras.optimizers.Adamax(
    learning_rate=l_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adamax"),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'])

train, test = create_dataframe()
historico = modelo_cnn.fit(train, epochs=epochs, validation_data=test)
modelo_cnn.save('/home/jhermida/Modelos/modelo12.h5')