import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras import Model
from keras.callbacks import Callback
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

class ValidationAccuracyStop(Callback):
    def __init__(self, accuracy=0.97):
        super(ValidationAccuracyStop, self).__init__()
        self.accuracy = accuracy

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('accuracy') >= self.accuracy and logs.get('val_accuracy') >= self.accuracy:
            print("\nAccuracy of %0.2f%% reached, stopping training." % (self.accuracy * 100))
            self.model.stop_training = True

callbacks = ValidationAccuracyStop()

pre_trained_model =  InceptionV3(input_shape=(150,150,3),
                                    include_top=False,
                                    weights=local_weights_file
                                    )

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer =  pre_trained_model.get_layer('mixed7')
outputLayer =  last_layer.output

train_dir = 'data/horse-or-human'
validation_dir = 'data/validation-horse-or-human'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.20,
    height_shift_range=0.20,
    shear_range=0.20,
    zoom_range=0.20,
    rotation_range=30,
    horizontal_flip=True,
    fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

x = layers.Flatten()(outputLayer)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer=RMSprop(lr=0.0001),
                loss='binary_crossentropy',
                metrics=['accuracy'])
model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[callbacks]
)