
from settings import IMG_DIM, COLOR_CHAN, NUM_CLASSES

from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense , Flatten , Conv2D , MaxPool2D, BatchNormalization, Dropout, AveragePooling2D



# ========================================================================================

def get_med_cnn():

    reg_para = 5
    assert 0 <= reg_para <= 5
    l = 10**(3.5-reg_para)
    drop_p = reg_para/10

    cnn = Sequential([
                        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN), kernel_regularizer=regularizers.l1_l2(l)),
                        BatchNormalization(),
                        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l1_l2(l)),
                        BatchNormalization(),
                        Conv2D(filters=32, kernel_size=(5, 5), strides=3, padding='same', activation='relu', kernel_regularizer=regularizers.l1_l2(l)),
                        BatchNormalization(),
                        Dropout(drop_p),

                        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l1_l2(l)),
                        BatchNormalization(),
                        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l1_l2(l)),
                        BatchNormalization(),
                        Conv2D(filters=64, kernel_size=(5, 5), strides=3, padding='same', activation='relu', kernel_regularizer=regularizers.l1_l2(l)),
                        BatchNormalization(),
                        Dropout(drop_p),

                        Conv2D(filters=128, kernel_size=(4, 4), activation='relu', kernel_regularizer=regularizers.l1_l2(l)),
                        BatchNormalization(),
                        Flatten(),
                        Dropout(drop_p),
                        Dense(10, activation='softmax', kernel_regularizer=regularizers.l1_l2(l))
                    ])

    return cnn

# ========================================================================================

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
                        Dense(32, activation='relu' ),
                        Dense(NUM_CLASSES, activation='softmax')
                      ])

# ========================================================================================

def get_dense_lowres_nn():

    l = 0.03

    dense_lowres_nn = Sequential([
                                    AveragePooling2D(pool_size=(3,3), input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN) ),
                                    Dropout(0.45),
                                    Flatten(),
                                    Dense(16, activation='relu', kernel_regularizer=regularizers.l2( l ) ),
                                    Dropout(0.4),
                                    Dense(16, activation='relu', kernel_regularizer=regularizers.l2(l)),
                                    Dropout(0.4),
                                    Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizers.l2( l))
                                 ])

    return dense_lowres_nn

# ========================================================================================



# ========================================================================================

def get_cnn_lowres():


    reg_para = 0
    assert 0 <= reg_para <= 5
    l = 10**(reg_para-6)
    drop_p = reg_para/10

    dense_lowres_nn = Sequential([
                                    Conv2D(filters=8, kernel_size=(5, 5), strides=3, padding='same', activation='relu',
                                           kernel_regularizer=regularizers.l1_l2(l),  input_shape=(IMG_DIM, IMG_DIM, COLOR_CHAN)  ),


                                    Conv2D(filters=16, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(l)),

                                    Flatten(),
                                    # Dropout(drop_p),

                                    Dense(8, activation='relu', kernel_regularizer=regularizers.l2( l ) ),

                                    # Dropout(drop_p),

                                    Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizers.l2( l))
                                 ])

    return dense_lowres_nn

# ========================================================================================