import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib

np.random.seed(123)
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

# import cv2
import os
from tqdm.keras import TqdmCallback
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

# from tensorflow.keras.engine.base_layer import Layer

from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Bottleneck_JQ(layers.Layer):
    """
    This class replicates the Bottleneck block of the FishNet modle

    This layer class takes in the concatenated outputs in the FishNet of the SAME DIMENSIONS and reduces the number of channels.

    There are two paths through this layer. The first is a series of convolutions, the second, optional path, is a channel reduciton funciton. The output from each path are added together to arrive at the final output

    inplanes is the number of input channels, planes is the number of output channels. In between the convolutions use bottleneck_channels, which are 1/4 of the number of output channels

    inplanes is implemented in pytorch but it is not necessary in keras. the iput channels is given by the previous layer (i think)

    The 'squeeze_idt' function has been renamed as the 'channel_reduction' although it can also increase the number of channels.

    k needs to be set as an integer. According to the paper, k is supposed to be c_in//c_out but this is something we need to double check.

    """

    def __init__(self, inplanes, planes, stride=1, mode="NORM", k=1, dilation=1):

        super(Bottleneck_JQ, self).__init__()

        self.stride = stride
        self.mode = mode
        self.k = k
        self.planes = planes

        bottleneck_channels = (
            planes // 4
        )  ##the number of channels in the bottlenect is 1/4 of the output channels. Need to read the paper to understand why

        self.relu = layers.Activation(
            "relu"
        )  # same relu activation settings is used in all parts of this class, so it is only defined once.

        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(
            filters=bottleneck_channels, kernel_size=1, activation=None, padding="same", use_bias=False
        )

        self.bn2 = (
            layers.BatchNormalization()
        )  # this is unecessary because the input channels are already specified by the previous layer in keras, but i'm keeping it for symetry with the original code
        self.conv2 = layers.Conv2D(
            filters=bottleneck_channels,
            kernel_size=3,
            strides=stride,
            use_bias=False,
            activation=None,
            padding="same",
            dilation_rate=dilation,
        )

        self.bn3 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(
            filters=planes, kernel_size=1, use_bias=False, activation=None, padding="same", dilation_rate=dilation
        )

        if mode == "UP":
            self.shortcut = None
        elif (
            inplanes != planes or stride > 1
        ):  # i don't know how to apply the inplanes != planes logic here... its not something that would be relevant to keras, but i guess we can also just forcibly specify it. UPDATE: that's exactly what i did, but its not elegant.
            self.shortcut = keras.Sequential()
            self.shortcut.add(layers.BatchNormalization())
            self.shortcut.add(layers.Activation("relu"))
            self.shortcut.add(
                layers.Conv2D(
                    filters=planes,
                    kernel_size=1,
                    strides=stride,
                    use_bias=False,
                    activation=None,
                    padding="same",
                    dilation_rate=dilation,
                )
            )
        else:
            self.shortcut = None  # if the inplanes equal the planes, ie no change in number of channels, then we just don't implement this bypass step

    def _pre_act_forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.mode == "UP":
            residual = self.channel_reduction(x)
        elif self.shortcut is not None:
            residual = self.shortcut(residual)

        out = layers.add([out, residual])

        return out

    def channel_reduction(self, idt):
        n = tf.shape(idt)[0]
        h = tf.shape(idt)[1]
        w = tf.shape(idt)[2]
        c = tf.shape(idt)[3]
        # n, h, w, c = tf.shape(idt)
        ## the above fails because these can be undefined when building the computational graph.
        ## see https://github.com/tensorflow/models/issues/6245
        idt_rehsaped = tf.reshape(idt, (n, h, w, c // self.k, self.k))
        idt_reduced = tf.math.reduce_sum(idt_rehsaped, axis=-1)
        return idt_reduced  # this should be correct and line 274 should be modified.

        # k = self.k
        # planes = self.planes
        # store = tf.math.reduce_sum(idt[:, :, :, 0:k], axis=3, keepdims=True)
        # for i in range(1, planes):
        #     # print(store.shape)
        #     store = layers.concatenate(
        #         [store, tf.math.reduce_sum(idt[:, :, :, i * k : (i * k) + k], axis=3, keepdims=True)]
        #     )
        #     # when planes and k don't agree, this ends up dropping the rest layers.
        #     # e.g. Bottleneck_JQ(256, 64, mode="UP")(x57)
        # return store

    def call(self, x):
        out = self._pre_act_forward(x)
        return out


# cifar10 shenanigans
# cifar = keras.Input(shape=(32, 32, 3))
# img_inputs2 = layers.UpSampling2D((7, 7))(cifar)

# what i actually wanted to train
img_inputs2 = keras.Input(shape=(224, 224, 3))

# 112X112 pre-Stage
x1 = layers.Conv2D(32, 3, activation=None, padding="same", strides=(2, 2), use_bias=False)(img_inputs2)
x2 = layers.BatchNormalization()(x1)
x3 = layers.Activation("relu")(x2)

x4 = layers.Conv2D(32, 3, activation=None, padding="same", strides=(1, 1), use_bias=False)(x3)
x5 = layers.BatchNormalization()(x4)
x6 = layers.Activation("relu")(x5)

x7 = layers.Conv2D(64, 3, activation=None, padding="same", strides=(1, 1), use_bias=False)(x6)
x8 = layers.BatchNormalization()(x7)
x9 = layers.Activation("relu")(x8)

x10 = layers.MaxPool2D(2, strides=2)(x9)  # first max pool -> Kernel Size 3

# 56x56 stage
x11 = Bottleneck_JQ(64, 128)(x10)  # NORM type bottleneck layer, shortcut is used.
x12 = Bottleneck_JQ(128, 128)(x11)  # NORM type but inplanes = planes so there will be no shortcut

x13 = layers.MaxPool2D(2, strides=2)(x12)

# 28X28 Stage
x14 = Bottleneck_JQ(128, 256)(x13)
x15 = Bottleneck_JQ(256, 256)(x14)
x16 = Bottleneck_JQ(256, 256)(x15)
x17 = Bottleneck_JQ(256, 256)(x16)

x17a = layers.MaxPool2D(2, strides=2)(
    x17
)  # this does not show up in the hierarchial summary table, but it can be found in the detailed code section.

# 14X14 Stage
x18 = Bottleneck_JQ(256, 512)(x17a)
x19 = Bottleneck_JQ(512, 512)(x18)
x20 = Bottleneck_JQ(512, 512)(x19)
x21 = Bottleneck_JQ(512, 512)(x20)
x22 = Bottleneck_JQ(512, 512)(x21)
x23 = Bottleneck_JQ(512, 512)(x22)
x24 = Bottleneck_JQ(512, 512)(x23)
x25 = Bottleneck_JQ(512, 512)(x24)

x25a = layers.MaxPool2D(2, strides=2)(x25)  # did not appear in summary

# 7X7 Stage
x26 = layers.BatchNormalization()(x25a)
x27 = layers.Activation("relu")(x26)
x28 = layers.Conv2D(256, 1, activation=None, padding="same", strides=(1, 1), use_bias=False)(x27)
x29 = layers.BatchNormalization()(x28)
x29 = layers.Activation("relu")(x29)  # did not appear in summary as well

x30 = layers.Conv2D(1024, 1, activation=None, padding="same", strides=(1, 1), use_bias=False)(x29)

x31 = Bottleneck_JQ(1024, 512)(x30)
x32 = Bottleneck_JQ(512, 512)(x31)
x33 = Bottleneck_JQ(512, 512)(x32)
x34 = Bottleneck_JQ(512, 512)(x33)

# SE Block
x35 = layers.BatchNormalization()(x34)
x36 = layers.Activation("relu")(x35)
x37 = layers.GlobalAveragePooling2D()(x36)  # does not translate from pytorch perfectly
x38 = layers.Reshape((1, 1, 512))(x37)
x39 = layers.Conv2D(32, 1, activation=None, padding="same", strides=(1, 1), use_bias=False)(x38)  # sq_conv
x40 = layers.Activation("relu")(x39)
x41 = layers.Conv2D(512, 1, activation=None, padding="same", strides=(1, 1), use_bias=False)(x40)  # ex_conv
x42 = layers.Activation("sigmoid")(x41)

# upsampling
x42a = layers.UpSampling2D((7, 7))(x42)

# BODY
# 7X7 Stage
x43 = Bottleneck_JQ(512, 512)(x42a)
x43a = Bottleneck_JQ(512, 512)(x43)
x43b = Bottleneck_JQ(512, 512)(x43a)
x43c = Bottleneck_JQ(512, 512)(x43b)
x43d = Bottleneck_JQ(512, 512)(x43c)
x44 = Bottleneck_JQ(512, 512)(x43d)

# upsampling
x45 = layers.UpSampling2D((2, 2))(x44)

# 14X14 Stage
x46 = Bottleneck_JQ(512, 256, mode="UP", k=2)(x45)  # no shortcut, uses channel reduction function.
x47 = Bottleneck_JQ(256, 256)(x46)
x48 = layers.Concatenate()([x47, x25])  # Concatenation does not happen at the begining of the stage?
x49 = Bottleneck_JQ(768, 384, mode="UP", k=2)(x48)
x50 = Bottleneck_JQ(384, 384)(x49)


# upsampling
x51 = layers.UpSampling2D((2, 2))(x50)

# 28x28 Stage
x52 = Bottleneck_JQ(384, 128, mode="UP", k=3)(x51)
x53 = Bottleneck_JQ(128, 128)(x52)
x53a = layers.UpSampling3D((1, 1, 2))(x53)  # upsampled the channels in order to make the numbers work
x54 = layers.Concatenate()([x53a, x17])  # Concatenation does not happen at the begining of the stage?
x55 = Bottleneck_JQ(512, 256, mode="UP", k=2)(x54)
x56 = Bottleneck_JQ(256, 256)(x55)

# upsampling
x57 = layers.UpSampling2D((2, 2))(x56)

# 56x56 Stage
x58 = Bottleneck_JQ(256, 64, mode="UP", k=4)(x57)
x59 = Bottleneck_JQ(64, 64)(x58)
x59a = layers.UpSampling3D((1, 1, 3))(x59)  # another weird upsampling situation to make the numbers work
x60 = layers.Concatenate()([x59a, x12])  # Concatenation does not happen at the begining of the stage?
x61 = Bottleneck_JQ(320, 320)(x60)  # THERE IS SOME SUPREME WEIRDNESS GOING ON HERE
x62 = Bottleneck_JQ(320, 320)(x61)


# downsampling
x63 = layers.MaxPool2D(2, strides=2)(x62)

# begining of head 28X28
# my suspicion, it that the transfered layer is run through 2 bottleneck layers before being concatenated.
x54_t1 = Bottleneck_JQ(512, 512)(x54)
x54_t2 = Bottleneck_JQ(512, 512)(x54_t1)

x64 = layers.Concatenate()([x63, x54_t2])  # this is the only way to get a layer with 832 filters
x65 = Bottleneck_JQ(832, 832)(x64)
x66 = Bottleneck_JQ(832, 832)(x65)

# downsampling
x67 = layers.MaxPool2D(2, strides=2)(x66)

# 14X14 head
# same thing happens here, the transfered block gets run through 2 bottleneck layers before being concatenated
x48_t1 = Bottleneck_JQ(768, 768)(x48)
x48_t2 = Bottleneck_JQ(768, 768)(x48_t1)

x68 = layers.Concatenate()([x67, x48_t2])  # again, this is the only way I know how to get a
x69 = Bottleneck_JQ(1600, 1600)(x68)
x70 = Bottleneck_JQ(1600, 1600)(x69)

# downsampling
x71 = layers.MaxPool2D(2, strides=2)(x70)

# 7X7 head
# same thing happens here, except this time the transfered block is run through 4 bottlenexk layers.
x44_t1 = Bottleneck_JQ(512, 512)(x44)
x44_t2 = Bottleneck_JQ(512, 512)(x44_t1)
x44_t3 = Bottleneck_JQ(512, 512)(x44_t2)
x44_t4 = Bottleneck_JQ(512, 512)(x44_t3)

x72 = layers.Concatenate()([x71, x44_t4])
x73 = layers.BatchNormalization()(x72)
x74 = layers.Activation("relu")(x73)
x75 = layers.Conv2D(1056, 1, activation=None, padding="same", strides=(1, 1), use_bias=False)(x74)
x76 = layers.BatchNormalization()(x75)
x77 = layers.GlobalAveragePooling2D()(x76)  # does not translate from pytorch perfectly
x78 = layers.Reshape((1, 1, 1056))(x77)
x79 = layers.Conv2D(1000, 1, activation=None, padding="same", strides=(1, 1), use_bias=False)(x78)
# OMG WE'RE FINALLY DONE

# output layers (these aren't part of FishNet, this is our own top layer to make it fit the dataset we are using)
img_outputs2 = layers.Flatten()(x79)
test_mdl2 = keras.Model(img_inputs2, img_outputs2, name="test_mdl4")
test_mdl2.summary()

opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)  # TODO:weight decay
test_mdl2.compile(
    optimizer=opt,
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
    run_eagerly=True,  # for debug
)

# data setup
train_datagen = ImageDataGenerator(horizontal_flip=True, rescale=1.0 / 255)
# train_path = "/root/imagenet/train"
train_path = "/root/imagenet/val"  # for test
train_it = train_datagen.flow_from_directory(train_path, target_size=(224, 224), shuffle=True, batch_size=32)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_path = "/root/imagenet/val"
val_it = val_datagen.flow_from_directory(val_path, target_size=(224, 224), shuffle=False, batch_size=32)

# fit
history = test_mdl2.fit(
    x=train_it,
    epochs=5,
    validation_data=val_it,
    callbacks=[TqdmCallback(verbose=1)],
)
hist = pd.DataFrame(history.history)

fig, ax = plt.subplots(1, 2, figsize=(15, 4))
hist.plot(y=["loss", "val_loss"], ax=ax[0])
hist.plot(y=["accuracy", "val_accuracy"], ax=ax[1])
