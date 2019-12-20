from get_data import create_data_with_labels
from display_utils import plot_image, plot_image_w_predictions, plot_value_array, plot_predictions_and_images, plot_labels_and_images

from settings import IMG_DIM, COLOR_CHAN, NUM_CLASSES

# TensorFlow and tf.keras
import tensorflow as tf

# fix tensorflow-gpu bug
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense , Flatten , Conv2D , MaxPool2D, BatchNormalization, Dropout
from tensorflow.keras.optimizers.schedules import ExponentialDecay






simpl_cnn = keras.Sequential([
                                Conv2D(filters=4, kernel_size=(3,3), activation='relu', input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN)),
                                BatchNormalization(),
                                MaxPool2D(pool_size=(3,3)),
                                Dropout(0.4),
                                Flatten(),
                                Dense(32, activation='relu'),
                                Dense(NUM_CLASSES, activation='softmax')
                             ])



simpl_dense_nn = keras.Sequential([
                                    Flatten(input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN) ),
                                    Dense(32, activation='relu', ),
                                    Dense(NUM_CLASSES, activation='softmax')
                             ])



if __name__ == "__main__":

    (train_images, train_labels) = create_data_with_labels("../data/train/")
    (test_images, test_labels) = create_data_with_labels("../data/test/")

    # print(train_labels)
    # print(test_labels)
    # plot_labels_and_images(train_images, train_labels)

    model = simpl_cnn

    print(model.summary())

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=ExponentialDecay(initial_learning_rate=0.001, decay_steps=60000, decay_rate=0.95)),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


    history = model.fit(train_images, train_labels, epochs=10, batch_size = 32,   validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

    numeric_predictions = model.predict(test_images)
    boolean_predictions = np.argmax(numeric_predictions, axis=1 )
    incorrect_mask = (boolean_predictions != test_labels)

    # plot_predictions_and_images(predictions=numeric_predictions[incorrect_mask], test_labels=test_labels[incorrect_mask], test_images=test_images[incorrect_mask])

    plot_predictions_and_images(predictions=numeric_predictions, test_labels=test_labels, test_images=test_images)