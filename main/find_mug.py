# TensorFlow
import tensorflow as tf

# fix tensorflow-gpu bug
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


from get_data import create_data_with_labels
from display_utils import plot_image, plot_image_w_predictions, plot_value_array, \
                          plot_predictions_and_images, plot_labels_and_images, show_flowed_data

from models import *
from data_augmentors import get_datagen_spec




import numpy as np
from tensorflow.keras.optimizers.schedules import ExponentialDecay



# =============================================
# =============================================




if __name__ == "__main__":

    (train_images, train_labels) = create_data_with_labels("../data/train/")
    (test_images, test_labels) = create_data_with_labels("../data/test/")

    #  show flowed data if desired
    if False: show_flowed_data(get_datagen_spec(transf_amnt = 1.0), train_images[0:1], train_labels[0:1])
    # show images and labels if desired
    if False: plot_labels_and_images(train_images, train_labels)


    model = get_med_cnn()
    print(model.summary())


    model.compile(optimizer=tf.optimizers.Adam(learning_rate=ExponentialDecay(initial_learning_rate=0.001, decay_steps=60000, decay_rate=0.95)),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


    batch_size = 64
    train_size = len(train_labels)
    steps_per_epoch = int(train_size / batch_size)
    print(steps_per_epoch)



    # history = model.fit(x=train_images , y=train_labels , epochs=25, validation_data=(test_images, test_labels))

    dgen_it = get_datagen_spec(transf_amnt=1.0).flow(train_images, train_labels, batch_size=batch_size)
    history = model.fit_generator( generator=dgen_it, 	steps_per_epoch = train_size/batch_size ,  epochs=25,   validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

    numeric_predictions = model.predict(test_images)
    boolean_predictions = np.argmax(numeric_predictions, axis=1 )
    incorrect_mask = (boolean_predictions != test_labels)

    # plot_predictions_and_images(predictions=numeric_predictions[incorrect_mask], test_labels=test_labels[incorrect_mask], test_images=test_images[incorrect_mask])

    plot_predictions_and_images(predictions=numeric_predictions, test_labels=test_labels, test_images=test_images)



