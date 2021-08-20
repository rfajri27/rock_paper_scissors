import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
physical_device = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_device[0], True)


def rps_model():
    TRAINING_DIR = "data/rps/"
    train_datagen = ImageDataGenerator(
        rescale=1 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                        batch_size=126,
                                                        class_mode='categorical',
                                                        target_size=(150, 150))

    VALIDATION_DIR = "data/rps-validation/"
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                                  batch_size=11,
                                                                  class_mode='categorical',
                                                                  target_size=(150, 150))

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy') >= 0.99 and logs.get('val_accuracy') >= 0.9):
                self.model.stop_training = True

    callbacks = myCallback()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        #tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    optimizer = tf.optimizers.RMSprop(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_generator,
              steps_per_epoch=20,
              epochs=50,
              validation_data=validation_generator,
              validation_steps=3,
              callbacks=[callbacks])

    return model

# Save model as .h5 file
if __name__ == '__main__':
    model = rps_model()
    model.save("model.h5")

