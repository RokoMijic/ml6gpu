
from settings import IMG_DIM, COLOR_CHAN, NUM_CLASSES

from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense , Flatten , Conv2D , MaxPooling2D, BatchNormalization, Dropout, AveragePooling2D, Cropping2D



def get_reg(reg_para):
    assert 0 <= reg_para <= 5
    l = 10**(-5 + reg_para)
    drop_p = reg_para/15

    print('************************************************************')
    print('dropout: %s' % drop_p)
    print('l2 reg: %s' % l)
    print('************************************************************')

    return {'l':l, 'drop_p': drop_p }







def get_simple_cnn(reg_para):

    l = get_reg(reg_para)['l']
    drop_p = get_reg(reg_para)['drop_p']

    simpl_cnn = Sequential([
                            Conv2D(
                                   filters=16, kernel_size=(3,3), activation='relu',
                                   input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN),
                                   kernel_regularizer=regularizers.l2(l)
                                  ),


                            MaxPooling2D(pool_size=(3,3)),

                            Dropout(drop_p),

                            Flatten(),

                            Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l)),

                            Dropout(drop_p),

                            Dense(NUM_CLASSES, activation='softmax')
                         ])

    return simpl_cnn


def get_dense_nn(reg_para=0.5):

    reg = get_reg(reg_para)
    l = reg['l']
    drop_p = reg['drop_p']


    dense_nn = Sequential([
                            Flatten(input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN) ),
                            Dropout(drop_p),
                            Dense(32, activation='relu', kernel_regularizer=regularizers.l2( l ) ),
                            Dense(NUM_CLASSES, activation='softmax')
                          ])

    return dense_nn


def get_dense_lowres_nn(reg_para):

    reg = get_reg(reg_para)
    l = reg['l']
    drop_p = reg['drop_p']

    dense_lowres_nn = Sequential([
                                    AveragePooling2D(pool_size=(3,3), input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN) ),
                                    Dropout(drop_p),
                                    Flatten(),
                                    Dense(12, activation='relu', kernel_regularizer=regularizers.l2( l ) ),
                                    Dropout(drop_p),
                                    Dense(10, activation='relu', kernel_regularizer=regularizers.l2( l )),
                                    Dropout(drop_p),
                                    Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizers.l2( l ))
                                 ])

    return dense_lowres_nn



def get_cnn_v_lowres(reg_para=0.75):

    reg = get_reg(reg_para)
    l = reg['l']
    drop_p = reg['drop_p']

    model = Sequential([
                        Conv2D(filters=12, kernel_size=(5, 5), strides=3, padding='same', activation='relu',
                               kernel_regularizer=regularizers.l1_l2(l),  input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN)  ),

                        Dropout(drop_p),

                        Conv2D(filters=6, kernel_size=(5, 5), strides=3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l)),

                        Dropout(drop_p),

                        Flatten(),

                        Dense(6, activation='relu', kernel_regularizer=regularizers.l2( l ) ),

                        Dropout(drop_p),

                        Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizers.l2( l))
                     ])

    return model




def get_cnn_lowres(reg_para=1.75):

    reg = get_reg(reg_para)
    l = reg['l']
    drop_p = reg['drop_p']

    model = Sequential([
                        Conv2D(filters=16, kernel_size=(5, 5), strides=3, padding='same', activation='relu',
                               kernel_regularizer=regularizers.l1_l2(l),  input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN)  ),

                        Dropout(drop_p),

                        Conv2D(filters=8, kernel_size=(5, 5), strides=3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l)),

                        Dropout(drop_p),

                        Flatten(),

                        Dense(8, activation='relu', kernel_regularizer=regularizers.l2( l ) ),

                        Dropout(drop_p),

                        Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizers.l2( l))
                     ])

    return model



def get_cnn_low_cropping(reg_para=1.5):

    reg = get_reg(reg_para)
    l = reg['l']
    drop_p = reg['drop_p']

    model = Sequential([

                        Cropping2D(cropping=8, input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN)  ),

                        Conv2D(filters=10, kernel_size=(5, 5), strides=3, padding='same', activation='relu', kernel_regularizer=regularizers.l1_l2(l),  ),

                        Dropout(drop_p),

                        Conv2D(filters=16, kernel_size=(5, 5), strides=3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l)),

                        Dropout(drop_p),

                        Flatten(),

                        Dense(8, activation='relu', kernel_regularizer=regularizers.l2( l ) ),

                        Dropout(drop_p),

                        Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizers.l2( l))
                     ])

    return model




def get_cnn_med_res(reg_para=2.75):

    l = get_reg(reg_para)['l']
    drop_p = get_reg(reg_para)['drop_p']

    model = Sequential([
                                    Conv2D(filters=32, kernel_size=(5, 5), strides=3, padding='same', activation='relu',
                                           kernel_regularizer=regularizers.l1_l2(l),  input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN)  ),

                                    Dropout(drop_p),

                                    Conv2D(filters=10, kernel_size=(5, 5), strides=3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l)),

                                    Dropout(drop_p),

                                    Flatten(),

                                    Dense(10, activation='relu', kernel_regularizer=regularizers.l2( l ) ),

                                    Dropout(drop_p),

                                    Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizers.l2( l))
                                 ])

    return model


def get_cnn_better_res(reg_para=3.75):

    reg = get_reg(reg_para)

    l = reg['l']
    drop_p = reg['drop_p']

    model = Sequential([
                        Conv2D(filters=8, kernel_size=(4, 4), strides=2, padding='same', activation='relu',
                               kernel_regularizer=regularizers.l1_l2(l),  input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN)  ),

                        Dropout(drop_p),

                        Conv2D(filters=10, kernel_size=(4, 4), strides=2, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l)),

                        Dropout(drop_p),

                        Conv2D(filters=12, kernel_size=(4, 4), strides=2, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l)),

                        Dropout(drop_p),

                        Conv2D(filters=16, kernel_size=(4, 4), strides=2, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l)),

                        Dropout(drop_p),

                        Conv2D(filters=28, kernel_size=(4, 4), strides=2, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l)),

                        Dropout(drop_p),

                        Flatten(),

                        Dense(48, activation='relu', kernel_regularizer=regularizers.l2( l ) ),

                        Dropout(drop_p),

                        Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizers.l2( l))
                     ])
    return model


def get_small_lenet_crop_cnn(reg_para=1.95):

    reg = get_reg(reg_para)

    l = reg['l']
    drop_p = reg['drop_p']

    cnn = Sequential([

                        Cropping2D(cropping=8, input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN)),

                        Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN), kernel_regularizer=regularizers.l2( l  )),

                        MaxPooling2D(pool_size=3, padding='valid'),

                        Dropout( drop_p ),

                        Conv2D(filters=16, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2( l )),

                        Dropout( drop_p ),

                        Conv2D(filters=16, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(l)),

                        MaxPooling2D(pool_size=3, padding='valid'),

                        Flatten(),

                        Dropout(  drop_p ),

                        Dense(30, activation='relu', kernel_regularizer=regularizers.l2(l)),

                        Dropout(  drop_p ),

                        Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizers.l2( l ))
                    ])

    return cnn



def get_med_lenet_crop_cnn(reg_para=1.95):

    reg = get_reg(reg_para)

    l = reg['l']
    drop_p = reg['drop_p']

    cnn = Sequential([

                        Cropping2D(cropping=8, input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN)),

                        Conv2D(filters=6, kernel_size=(3, 3), activation='relu', padding='same',  input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN), kernel_regularizer=regularizers.l2( l  )),
                        MaxPooling2D(pool_size=3, padding='same'),

                        Dropout( drop_p ),

                        Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2( l )),
                        MaxPooling2D(pool_size=3, padding='same'),

                        Dropout( drop_p ),

                        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',  kernel_regularizer=regularizers.l2(l)),
                        MaxPooling2D(pool_size=3, padding='same'),

                        Flatten(),

                        Dropout(  drop_p ),

                        Dense(40, activation='relu', kernel_regularizer=regularizers.l2(l)),

                        Dropout(  drop_p ),

                        Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizers.l2( l ))
                    ])
    return cnn




def get_bigish_lenet_crop_cnn(reg_para=2.25):

    # This is the best model!

    reg = get_reg(reg_para)

    l = reg['l']
    drop_p = reg['drop_p']

    cnn = Sequential([

                        Cropping2D(cropping=9, input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN)),

                        Conv2D(filters=12, kernel_size=(3, 3), activation='relu', padding='same',  input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN), kernel_regularizer=regularizers.l2( l  )),
                        MaxPooling2D(pool_size=3, padding='same'),

                        Dropout( drop_p ),

                        Conv2D(filters=20, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2( l )),
                        MaxPooling2D(pool_size=3, padding='same'),

                        Dropout( drop_p ),

                        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',  kernel_regularizer=regularizers.l2(l)),
                        MaxPooling2D(pool_size=3, padding='same'),

                        Flatten(),

                        Dropout(  drop_p ),

                        Dense(30, activation='relu', kernel_regularizer=regularizers.l2(l)),

                        Dropout(  drop_p ),

                        Dense(15, activation='relu', kernel_regularizer=regularizers.l2(l)),

                        Dropout(drop_p),

                        Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizers.l2(l))
                    ])
    return cnn



def get_big_lenet_crop_cnn(reg_para=2.5):

    reg = get_reg(reg_para)

    l = reg['l']
    drop_p = reg['drop_p']

    cnn = Sequential([

                        Cropping2D(cropping=9, input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN)),

                        Conv2D(filters=10, kernel_size=(3, 3), activation='relu', padding='same',  input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN), kernel_regularizer=regularizers.l2( l  )),
                        MaxPooling2D(pool_size=3, padding='same'),

                        Dropout( drop_p ),

                        Conv2D(filters=20, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2( l )),
                        MaxPooling2D(pool_size=3, padding='same'),

                        Dropout( drop_p ),

                        Conv2D(filters=40, kernel_size=(3, 3), activation='relu', padding='same',  kernel_regularizer=regularizers.l2(l)),
                        MaxPooling2D(pool_size=3, padding='same'),

                        Flatten(),

                        Dropout(  drop_p ),

                        Dense(30, activation='relu', kernel_regularizer=regularizers.l2(l)),

                        Dropout(  drop_p ),

                        Dense(15, activation='relu', kernel_regularizer=regularizers.l2(l)),

                        Dropout(drop_p),

                        Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizers.l2(l))
                    ])
    return cnn



def get_huge_lenet_crop_cnn(reg_para=2.95):

    reg = get_reg(reg_para)

    l = reg['l']
    drop_p = reg['drop_p']

    cnn = Sequential([

                        Cropping2D(cropping=9, input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN)),

                        Conv2D(filters=20, kernel_size=(3, 3), activation='relu', padding='same',  input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN), kernel_regularizer=regularizers.l2( l  )),
                        MaxPooling2D(pool_size=3, padding='same'),

                        Dropout( drop_p ),

                        Conv2D(filters=35, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2( l )),
                        MaxPooling2D(pool_size=3, padding='same'),

                        Dropout( drop_p ),

                        Conv2D(filters=80, kernel_size=(3, 3), activation='relu', padding='same',  kernel_regularizer=regularizers.l2(l)),
                        MaxPooling2D(pool_size=3, padding='same'),

                        Flatten(),

                        Dropout(  drop_p ),

                        Dense(65, activation='relu', kernel_regularizer=regularizers.l2(l)),

                        Dropout(  drop_p ),

                        Dense(45, activation='relu', kernel_regularizer=regularizers.l2(l)),

                        Dropout(drop_p),

                        Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizers.l2(l))
                    ])
    return cnn


def get_lenet_cnn(reg_para=2.5):

    l = get_reg(reg_para)['l']
    drop_p = get_reg(reg_para)['drop_p']

    cnn = Sequential([
                        Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN), kernel_regularizer=regularizers.l2( l  )),

                        Conv2D(filters=8, kernel_size=(5, 5), strides=3, padding='same', activation='relu', kernel_regularizer=regularizers.l2( l )),

                        Dropout( drop_p ),

                        Conv2D(filters=10, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2( l )),

                        Conv2D(filters=14, kernel_size=(5, 5), strides=3, padding='same', activation='relu', kernel_regularizer=regularizers.l2( l )),

                        Dropout( drop_p ),

                        Conv2D(filters=16, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(l)),

                        Conv2D(filters=20, kernel_size=(5, 5), strides=3, padding='same', activation='relu',  kernel_regularizer=regularizers.l2(l)),

                        Flatten(),

                        Dropout(  drop_p ),

                        Dense(24, activation='relu', kernel_regularizer=regularizers.l2(l)),

                        Dropout(  drop_p ),

                        Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizers.l2( l ))
                    ])

    return cnn


def get_big_lenet_cnn(reg_para=2.95):

    l = get_reg(reg_para)['l']
    drop_p = get_reg(reg_para)['drop_p']

    cnn = Sequential([
                        Conv2D(filters=12, kernel_size=(3, 3), activation='relu', input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN), kernel_regularizer=regularizers.l2( l  )),

                        Conv2D(filters=16, kernel_size=(5, 5), strides=3, padding='same', activation='relu', kernel_regularizer=regularizers.l2( l )),

                        Dropout( drop_p ),

                        Conv2D(filters=20, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2( l )),

                        Conv2D(filters=28, kernel_size=(5, 5), strides=3, padding='same', activation='relu', kernel_regularizer=regularizers.l2( l )),

                        Dropout( drop_p ),

                        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(l)),

                        Conv2D(filters=40, kernel_size=(5, 5), strides=3, padding='same', activation='relu',  kernel_regularizer=regularizers.l2(l)),

                        Flatten(),

                        Dropout(  drop_p ),

                        Dense(48, activation='relu', kernel_regularizer=regularizers.l2(l)),

                        Dropout(  drop_p ),

                        Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizers.l2( l ))
                    ])

    return cnn





if __name__ == "__main__":

    model = get_bigish_lenet_crop_cnn()
    print(model.summary())

