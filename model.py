import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
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

    optimizer = tf.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    epoch = 50
    H = model.fit(train_generator,
              steps_per_epoch=20,
              epochs=epoch,
              validation_data=validation_generator,
              validation_steps=3)

    return model, H, epoch

# Save model as .h5 file
if __name__ == '__main__':
    model, H, epoch = rps_model()
    model.save("model.h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epoch), H.history["loss"], label="training")
plt.plot(np.arange(0, epoch), H.history["val_loss"], label="validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("images/plot.png")

plt.figure()
plt.plot(np.arange(0, epoch), H.history["accuracy"], label="training")
plt.plot(np.arange(0, epoch), H.history["val_accuracy"], label="validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("images/plot_acc.png")



