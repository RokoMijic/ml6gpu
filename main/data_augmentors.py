
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# required for some preprocessing
import scipy as sp
import PIL





def get_datagen_spec(transf_amnt = 0.0):

    return  ImageDataGenerator(
                                rotation_range=20*transf_amnt,
                                width_shift_range=0.025*transf_amnt,
                                height_shift_range=0.025*transf_amnt,
                                shear_range=0.05*transf_amnt,
                                zoom_range=0.25*transf_amnt,
                                horizontal_flip=True,
                                channel_shift_range=0.09*transf_amnt,
                                fill_mode='nearest'
                              )


if __name__ == "__main__":

    pass

    # save augmented images here