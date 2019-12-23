# TensorFlow
import tensorflow as tf

from uncertainties import ufloat

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





def train_model(p_model, p_train_images, p_train_labels, p_val_images, p_val_labels):

    p_model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'], )

    _ = p_model.fit(x=p_train_images, y=p_train_labels, epochs=70, validation_data=(p_val_images, p_val_labels), batch_size=512, shuffle=True)

    return p_model


def evaluate_model(p_model, images, labels):
    _ , val_acc_nominal = p_model.evaluate(images, labels, verbose=0)
    val_acc = ufloat(val_acc_nominal, ((val_acc_nominal*(1-val_acc_nominal)) / len(labels)) ** 0.5)
    return val_acc



def get_model_performance_estimate(p_model_fn, p_train_images, p_train_labels, p_val_images, p_val_labels,  p_test_images, p_test_labels):

    num_models = 3

    models = [None]*num_models
    val_accs = [None]*num_models
    test_accs = [None] * num_models
    for i in range(num_models):

        models[i] = train_model(p_model=p_model_fn(), p_train_images=p_train_images, p_train_labels=p_train_labels, p_val_images=p_val_images, p_val_labels=p_val_labels)
        val_accs[i] = evaluate_model(models[i], images=p_val_images, labels=p_val_labels)
        test_accs[i] = evaluate_model(models[i], images=p_test_images, labels=p_test_labels)

    average_accs = ( sum(val_accs) + sum(test_accs)  ) / ( len(val_accs) + len(test_accs) )

    print('Estimated accuracy of model population on val and test data is %s' % average_accs)

    return models


def predict_from_models(p_models, p_images):


    boolean_predictions_list = []
    numeric_predictions_list = []

    for model in p_models:

        numeric_predictions = model.predict(p_images)
        numeric_predictions_list.append(numeric_predictions)

        boolean_predictions = np.argmax(numeric_predictions, axis=1)
        boolean_predictions_list.append(boolean_predictions)


    hard_ensemble_preds = np.array([ max(set(p), key=p.count).item()  for p in list(zip( *[l for l in boolean_predictions_list] )) ])

    ensembled_probs  =  [  list(np.mean(p, axis=0) ) for p in list(zip( *[l for l in numeric_predictions_list] )) ]

    average_ensemble_preds = np.array([  np.argmax(p) for p in  ensembled_probs ])

    return { 'hard_ensemble_preds': hard_ensemble_preds, 'average_ensemble_preds': average_ensemble_preds, 'ensembled_probs': ensembled_probs }





if __name__ == "__main__":

    (train_images, train_labels) = create_data_with_labels("../data_aug3/train/")
    # (train_images, train_labels) = create_data_with_labels("../data/train/")
    (t_val_images, t_val_labels) = create_data_with_labels("../data/test/")
    (test_images, test_labels) = (t_val_images[0:1000 ] , t_val_labels[0:1000 ]  )
    (val_images, val_labels) = (t_val_images[ 0:1000], t_val_labels[0:1000 ])

    model_fn = get_bigish_lenet_crop_cnn

    models = get_model_performance_estimate(p_model_fn=model_fn, p_train_images=train_images, p_train_labels=train_labels,
                                                                     p_val_images=val_images, p_val_labels=val_labels,
                                                                     p_test_images=test_images, p_test_labels=test_labels
                                                )



    ensemble_probabilities = predict_from_models(models, test_images )['ensembled_probs']
    ensemble_predictions =   predict_from_models(models, test_images )['average_ensemble_preds']
    correct_preds = ensemble_predictions == test_labels
    ensemble_accuracy = sum(correct_preds) / len(test_labels)
    ensemble_accuracy_ufloat = ufloat(ensemble_accuracy, ((ensemble_accuracy * (1 - ensemble_accuracy)) / len(test_labels)) ** 0.5)

    print( 'ensemble accuracy is %s ' % ensemble_accuracy_ufloat)

    predicted_black = (ensemble_predictions == 1)
    is_black = (test_labels == 1)
    correct_black = (predicted_black == is_black)
    black_accuracy = sum(correct_black)/len(test_labels)

    print("Ensemble accuracy on black mug = %s" % black_accuracy)



    incorrect_mask = ensemble_predictions != test_labels
    incorrect_probs = [pl for (pl, m) in zip(ensemble_probabilities, incorrect_mask) if m]
    incorrect_labels = test_labels[incorrect_mask]
    incorrect_images = test_images[incorrect_mask]

    plot_predictions_and_images(predictions=incorrect_probs, test_labels= incorrect_labels  , test_images=incorrect_images)

    plot_predictions_and_images(predictions=ensemble_probabilities, test_labels=test_labels, test_images=test_images)


