# TensorFlow
import tensorflow as tf

from uncertainties import ufloat

# fix tensorflow-gpu bug
from settings import CLASS_NAMES

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


from get_data import create_data_with_labels
from display_utils import plot_image, plot_image_w_predictions, plot_value_array, \
    plot_preds_imgs, plot_labels_and_images, show_flowed_data, plot_preds_imgs_masked

from models import *
from data_augmentors import get_datagen_w_transforms


import numpy as np
from tensorflow.keras.optimizers.schedules import ExponentialDecay



# =============================================
# =============================================





def train_model(p_model_fn, p_train_images, p_train_labels, p_val_images, p_val_labels):

    p_model = p_model_fn()

    p_model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'], )

    _ = p_model.fit(x=p_train_images, y=p_train_labels, epochs=60, validation_data=(p_val_images, p_val_labels), batch_size=512, shuffle=True)

    return p_model


def evaluate_model(p_model, images, labels):
    _ , val_acc_nominal = p_model.evaluate(images, labels, verbose=0)
    val_acc = ufloat(val_acc_nominal, ((val_acc_nominal*(1-val_acc_nominal)) / len(labels)) ** 0.5)
    return val_acc




if __name__ == "__main__":

    (train_images, train_labels) = create_data_with_labels("../data_aug3/train/")
    # (train_images, train_labels) = create_data_with_labels("../data/train/")
    (test_images, test_labels) = create_data_with_labels("../data/test/")


    model_fn = get_bigish_lenet_crop_cnn

    trained_model = train_model(p_model_fn=model_fn, p_train_images=train_images, p_train_labels=train_labels, p_val_images=test_images, p_val_labels=test_labels)

    prob_preds = trained_model.predict(test_images)
    label_preds = np.argmax(prob_preds, axis=1)

    # print(label_preds)


    accuracy = sum(label_preds == test_labels) / len(test_labels)
    accuracy_ufloat = ufloat(accuracy, 0.02)

    print( 'accuracy is %s ' % accuracy_ufloat)

    for i,c in enumerate(CLASS_NAMES):
        predicted_class_i = (label_preds == i)
        is_i = (test_labels == i)
        correct_class_i = (predicted_class_i == is_i)
        class_i_accuracy = sum(correct_class_i)/len(test_labels)

        print("Accuracy on %s mug = %s" % (c , class_i_accuracy) )



    for i, _ in enumerate(CLASS_NAMES):
        print(i)

        plot_preds_imgs_masked(probs=prob_preds, test_labels=test_labels, test_images=test_images, mask=(test_labels == i) & (label_preds == test_labels))
        plot_preds_imgs_masked(probs=prob_preds, test_labels=test_labels, test_images=test_images, mask=(test_labels == i) & (label_preds != test_labels))








