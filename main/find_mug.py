# TensorFlow
import tensorflow as tf

# fix tensorflow-gpu bug
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


from get_data import create_data_with_labels
from display_utils import plot_image, plot_image_w_predictions, plot_value_array, \
                          plot_predictions_and_images, plot_labels_and_images, show_flowed_data

from models import *
from data_augmentors import get_datagen_w_transforms




import numpy as np
from tensorflow.keras.optimizers.schedules import ExponentialDecay



# =============================================
# =============================================




if __name__ == "__main__":

    (train_images, train_labels) = create_data_with_labels("../data_aug3/train/")
    # (train_images, train_labels) = create_data_with_labels("../data_augmented/train/")
    # (train_images, train_labels) = create_data_with_labels("../data/train/")
    (test_images, test_labels) = create_data_with_labels("../data/test/")

    if False: show_flowed_data(get_datagen_spec(transf_amnt = 1.0), train_images[0:1], train_labels[0:1])
    if False: plot_labels_and_images(train_images, train_labels)
    if False: plot_labels_and_images(test_images, test_labels)


    model = get_cnn_v_lowres(reg_para=0.75)
    print(model.summary())

    # # optional decaying learning rate
    # learning_rate = ExponentialDecay(initial_learning_rate=0.001, decay_steps=20000, decay_rate=0.80)


    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


    batch_size = 1024
    # train_size = len(train_labels)
    # steps_per_epoch = int(train_size / batch_size)
    # print(steps_per_epoch)


    history = model.fit(x=train_images , y=train_labels , epochs=300, validation_data=(test_images, test_labels), batch_size=batch_size  )

    # dgen_it = get_datagen_spec(transf_amnt=1.0).flow(train_images, train_labels, batch_size=batch_size)
    # history = model.fit_generator( generator=dgen_it, 	steps_per_epoch = train_size/batch_size ,  epochs=25,   validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

    numeric_predictions = model.predict(test_images)
    boolean_predictions = np.argmax(numeric_predictions, axis=1 )
    predicted_black = (boolean_predictions == 1)

    is_black = test_labels == 1

    # print(is_black)
    # print(predicted_black)

    correct_black = (predicted_black == is_black)
    black_accuracy = sum(correct_black)/len(test_labels)

    print("Accuracy on black mug = %s" % black_accuracy)




    #
    #
    # incorrect_mask = (boolean_predictions != test_labels)
    #
    # # plot_predictions_and_images(predictions=numeric_predictions[incorrect_mask], test_labels=test_labels[incorrect_mask], test_images=test_images[incorrect_mask])
    #
    # plot_predictions_and_images(predictions=numeric_predictions, test_labels=test_labels, test_images=test_images)
    #
    #
    #
