import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPool2D, BatchNormalization, ReLU, Add, \
    Concatenate, ELU, AveragePooling2D
from tensorflow.keras.models import Model


###############################
#         IRUNet MODEL        #
###############################


def inceptionBlock(inputVector,filters):
    branch_a = Conv2D(filters,   (3, 3), strides=1, padding='same',activation='relu', kernel_initializer="he_uniform")(inputVector)
    branch_b = Conv2D(filters*2, (3, 3), strides=1, padding='same',activation='relu', kernel_initializer="he_uniform")(inputVector)
    branch_c = Conv2D(filters,   (3, 3), strides=1, padding='same',activation='relu', kernel_initializer="he_uniform",dilation_rate=2)(inputVector)

    concat   = Concatenate()([branch_a,branch_b,branch_c])
    filter_reduction = Conv2D(filters,(1, 1), strides=1, padding='same')(concat)
    shortcut = Add()([inputVector,filter_reduction])
    return shortcut

def inceptionBlockReduction(inputVector,filters):
    shortcut = Conv2D(filters,   (2, 2), strides=2, padding='same')(inputVector)
    branch_a = Conv2D(filters,   (3, 3), strides=2, padding='same',activation='relu', kernel_initializer="he_uniform")(inputVector)
    branch_b = Conv2D(filters*2, (3, 3), strides=2, padding='same',activation='relu', kernel_initializer="he_uniform")(inputVector)
    branch_c = AveragePooling2D((2,2), padding='same')(inputVector)

    concat   = Concatenate()([branch_a,branch_b,branch_c])
    filter_reduction = Conv2D(filters,(1, 1), strides=1, padding='same')(concat)
    shortcut = Add()([shortcut,filter_reduction])
    return shortcut



def IRUNet(H, W, C, filters):
    inputs = Input(shape=(H, W, C))
    input_image = inputs


    head    = Conv2D(filters, (3, 3), strides=1, padding='same',activation='relu', kernel_initializer="he_uniform")(input_image)

    conv1   = inceptionBlockReduction(head,filters)
    conv1   = inceptionBlock(conv1,filters)

    conv2   = inceptionBlockReduction(conv1,filters)
    conv2   = inceptionBlock(conv2,filters)

    conv3   = inceptionBlockReduction(conv2,filters)
    conv3   = inceptionBlock(conv3,filters)


    body    = inceptionBlockReduction(conv3,filters)
    body    = inceptionBlock(body,filters)


    deconv3 = Conv2DTranspose(filters, (2, 2), strides=2, padding='same',activation='relu', kernel_initializer="he_uniform")(body)
    deconv3 = inceptionBlock(deconv3,filters)
    #deconv3 = Add()([conv3,deconv3])

    deconv2 = Conv2DTranspose(filters, (2, 2), strides=2, padding='same',activation='relu', kernel_initializer="he_uniform")(deconv3)
    deconv2 = inceptionBlock(deconv2,filters)
    deconv2 = Add()([conv2,deconv2])

    deconv1 = Conv2DTranspose(filters, (2, 2), strides=2, padding='same',activation='relu', kernel_initializer="he_uniform")(deconv2)
    deconv1 = inceptionBlock(deconv1,filters)
    deconv1 = Add()([conv1,deconv1])
    

    tail    = Conv2DTranspose(filters, (2, 2), strides=2, padding='same',activation='relu', kernel_initializer="he_uniform")(deconv1)
    tail    = inceptionBlock(tail,filters)
    tail    = Conv2D(3,(1, 1), strides=1, padding='same', activation='sigmoid')(tail)
    

    return Model(inputs, tail)