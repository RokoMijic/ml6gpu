import itertools

import matplotlib.pyplot as plt
import numpy as np
from settings import CLASS_NAMES, IMG_DIM, COLOR_CHAN, NUM_CLASSES



def plot_image(img, label = '?'):

    plt.imshow(img.reshape(IMG_DIM, IMG_DIM, COLOR_CHAN))
    plt.xlabel(label)



def plot_image_w_predictions(i, predictions_array, true_label, img, class_names=CLASS_NAMES):

  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img.reshape(IMG_DIM, IMG_DIM, COLOR_CHAN), cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)



def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(NUM_CLASSES))
  plt.yticks([])
  thisplot = plt.bar(range(NUM_CLASSES), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# ( lambda _ : plot_preds_imgs(probs=prob_preds[ _ ], test_labels=test_labels[ _ ], test_images=test_images[ _ ])  ) (label_preds != test_labels)

def plot_preds_imgs_masked(probs, test_labels, test_images, mask, class_names=CLASS_NAMES):
    return plot_preds_imgs(probs=probs[ mask ], test_labels=test_labels[ mask ], test_images=test_images[ mask ], class_names=class_names)

def plot_preds_imgs(probs, test_labels, test_images, class_names=CLASS_NAMES):

    assert( len(probs) ==  len(test_labels) ==  len(test_images) )

    num_rows = 9
    num_cols = 7
    num_images = min( num_rows * num_cols,  len(probs))
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range( num_images ) :
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image_w_predictions(i, probs[i], test_labels, test_images, class_names=class_names)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, probs[i], test_labels)
    plt.tight_layout()
    plt.show()


def plot_labels_and_images(images, labels):

    num_rows = 5
    num_cols = 5
    num_images = 2 * num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols,  i + 1)
        plot_image(img=images[i], label=labels[i])


    plt.tight_layout()
    plt.show()


def show_flowed_data(datagen, images, labels ):
    it = datagen.flow(images, labels, batch_size=1)
    flowed_images, flowed_labels = zip(*( list(itertools.islice(it, 50)) ) )
    plot_labels_and_images(images=flowed_images , labels= flowed_labels )


