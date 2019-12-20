import itertools

import matplotlib.pyplot as plt

from get_data import create_data_with_labels
from display_utils import plot_image, plot_image_w_predictions, plot_value_array, plot_predictions_and_images, plot_labels_and_images, show_flowed_data

from settings import IMG_DIM, COLOR_CHAN, NUM_CLASSES

# TensorFlow
import tensorflow as tf

# fix tensorflow-gpu bug
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import numpy as np
import scipy as sp
import PIL

from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense , Flatten , Conv2D , MaxPool2D, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.schedules import ExponentialDecay


def get_simple_cnn():

    l2_l = 30.0
    drop_p = 0.4

    simpl_cnn = Sequential([
                            Conv2D(
                                   filters=16, kernel_size=(3,3), activation='relu',
                                   input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN),
                                   kernel_regularizer=regularizers.l2(l2_l)
                                  ),
                            BatchNormalization(),
                            MaxPool2D(pool_size=(3,3)),
                            Dropout(drop_p),
                            Flatten(),
                            Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_l)),
                            Dropout(drop_p),
                            Dense(NUM_CLASSES, activation='softmax')
                         ])
    
    return simpl_cnn


dense_nn = Sequential([
                        Flatten(input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN) ),
                        Dense(32, activation='relu', ),
                        Dense(NUM_CLASSES, activation='softmax')
                      ])


def get_datagen_spec(transf_amnt = 0.0):

    return  ImageDataGenerator(
                                rotation_range=20*transf_amnt,
                                width_shift_range=0.025*transf_amnt,
                                height_shift_range=0.025*transf_amnt,
                                shear_range=0.05*transf_amnt,
                                zoom_range=0.2*transf_amnt,
                                horizontal_flip=False,
                                channel_shift_range=0.1*transf_amnt,
                                fill_mode='nearest'
                              )



if __name__ == "__main__":

    (train_images, train_labels) = create_data_with_labels("../data/train/")
    (test_images, test_labels) = create_data_with_labels("../data/test/")

    #  show flowed data if desired
    if False: show_flowed_data(get_datagen_spec(transf_amnt = 1.0), train_images[0:5], train_labels[0:5])

    # show images and labels if desired
    if False: plot_labels_and_images(train_images, train_labels)

    model = get_simple_cnn()
    # model = dense_nn

    print(model.summary())



    model.compile(optimizer=tf.optimizers.Adam(learning_rate=ExponentialDecay(initial_learning_rate=0.001, decay_steps=60000, decay_rate=0.95)),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


    batch_size = 128
    train_size = len(train_labels)
    steps_per_epoch = int(train_size / batch_size)
    print(steps_per_epoch)

    dgen_it = get_datagen_spec(transf_amnt = 0.0).flow(train_images, train_labels, batch_size=batch_size)

    history = model.fit_generator( generator=dgen_it, 	steps_per_epoch = train_size/batch_size ,  epochs=25,   validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

    numeric_predictions = model.predict(test_images)
    boolean_predictions = np.argmax(numeric_predictions, axis=1 )
    incorrect_mask = (boolean_predictions != test_labels)

    # plot_predictions_and_images(predictions=numeric_predictions[incorrect_mask], test_labels=test_labels[incorrect_mask], test_images=test_images[incorrect_mask])
    #
    plot_predictions_and_images(predictions=numeric_predictions, test_labels=test_labels, test_images=test_images)



