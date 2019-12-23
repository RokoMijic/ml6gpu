from itertools import islice
import os

from settings import IMG_DIM, CLASS_NAMES
from data_augmentors import get_datagen_w_transforms
from get_data import create_data_with_labels



if __name__ == "__main__":

    assert False  # safety catch!

    SOURCE_DIR = '../data/train/'
    TARGET_DIR = '../data_aug3/train/'


    DIR_LOAD_FROM = SOURCE_DIR

    assert(os.path.isdir(DIR_LOAD_FROM))

    num_to_save_per_class = 4000


    for dclass in CLASS_NAMES:

        DIR_TO_SAVE_TO = os.path.join(TARGET_DIR, dclass)
        print(DIR_TO_SAVE_TO)
        assert (os.path.isdir(DIR_TO_SAVE_TO))

        i = 0
        for batch in get_datagen_w_transforms( transf_amnt=1.0 ).flow_from_directory(directory= DIR_LOAD_FROM ,
                                                                    target_size=(IMG_DIM, IMG_DIM),
                                                                    classes=[dclass],
                                                                    class_mode=None,
                                                                    shuffle=True,
                                                                    batch_size=1,
                                                                    save_to_dir=DIR_TO_SAVE_TO,
                                                                    save_format='jpg',
                                                                    save_prefix='gen'):

            i += 1
            if i >= num_to_save_per_class:
                break  # otherwise the generator would loop indefinitely




