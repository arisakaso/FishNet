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
from tensorflow.keras.callbacks import LearningRateScheduler

# from tensorflow.keras.engine.base_layer import Layer

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import wandb
from wandb.keras import WandbCallback
import tensorflow_addons as tfa


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

        idt_rehsaped = tf.reshape(idt, (n, h, w, c // self.k, self.k))
        idt_reduced = tf.math.reduce_sum(idt_rehsaped, axis=-1)
        return idt_reduced  # this should be correct and line 274 should be modified.

    # n, h, w, c = tf.shape(idt)
    ## the above fails because these can be undefined when building the computational graph.
    ## see https://github.com/tensorflow/models/issues/6245
    # k = self.k
    # planes = self.planes
    # store = tf.math.reduce_sum(idt[:,:,:,0:k],axis=3, keepdims=True)
    # for i in range(1,planes):
    #   print(store.shape)
    #   store = layers.concatenate([store,tf.math.reduce_sum(idt[:,:,:,i*k:(i*k)+k],axis=3, keepdims=True)])
    # return store

    def call(self, x):
        out = self._pre_act_forward(x)
        return out


number_classes = 18  # 1000 for Imagenet
run = wandb.init(project="fishnet")
# cifar10 shenanigans
# cifar = keras.Input(shape=(32, 32, 3))
# img_inputs2 = layers.UpSampling2D((7, 7))(cifar)

img_inputs = keras.Input(shape=(224, 224, 3))

# Start of keras model
x1 = layers.Conv2D(32, 3, activation=None, padding="same", strides=(2, 2), use_bias=False)(img_inputs)
x2 = layers.BatchNormalization()(x1)
x3 = layers.Activation("relu")(x2)

x4 = layers.Conv2D(32, 3, activation=None, padding="same", strides=(1, 1), use_bias=False)(x3)
x5 = layers.BatchNormalization()(x4)
x6 = layers.Activation("relu")(x5)

x7 = layers.Conv2D(64, 3, activation=None, padding="same", strides=(1, 1), use_bias=False)(x6)
x8 = layers.BatchNormalization()(x7)
x9 = layers.Activation("relu")(x8)

x10 = layers.MaxPool2D(2, strides=2)(x9)

# 56x56 stage
x11 = Bottleneck_JQ(64, 128)(x10)  # NORM type bottleneck layer, shortcut is used.
x12 = Bottleneck_JQ(128, 128)(x11)  # NORM type but inplanes = planes so there will be no shortcut

x13 = layers.MaxPool2D(2, strides=2)(x12)

# 28X28 Stage
x14 = Bottleneck_JQ(128, 256)(x13)
x15 = Bottleneck_JQ(256, 256)(x14)
x16 = Bottleneck_JQ(256, 256)(x15)
x17 = Bottleneck_JQ(256, 256)(x16)

x17a = layers.MaxPool2D(2, strides=2)(x17)

# 14X14 Stage
x18 = Bottleneck_JQ(256, 512)(x17a)
x19 = Bottleneck_JQ(512, 512)(x18)
x20 = Bottleneck_JQ(512, 512)(x19)
x21 = Bottleneck_JQ(512, 512)(x20)
x22 = Bottleneck_JQ(512, 512)(x21)
x23 = Bottleneck_JQ(512, 512)(x22)
x24 = Bottleneck_JQ(512, 512)(x23)
x25 = Bottleneck_JQ(512, 512)(x24)

x25a = layers.MaxPool2D(2, strides=2)(x25)

# 7X7 Stage
x26 = layers.BatchNormalization()(x25a)
x27 = layers.Activation("relu")(x26)
x28 = layers.Conv2D(256, 1, activation=None, padding="same", strides=(1, 1), use_bias=False)(x27)
x29 = layers.BatchNormalization()(x28)
x30 = layers.Conv2D(1024, 1, activation=None, padding="same", strides=(1, 1), use_bias=False)(x29)
x31 = layers.BatchNormalization()(x30)
x32 = layers.Activation("relu")(x31)

# 1X1 Stage
x33 = layers.GlobalAveragePooling2D()(x32)
x33a = layers.Reshape((1, 1, 1024))(x33)
x34 = layers.Conv2D(32, 1, activation=None, padding="same", strides=(1, 1), use_bias=False)(x33a)  # sq_conv
x35 = layers.Activation("relu")(x34)
x36 = layers.Conv2D(512, 1, activation=None, padding="same", strides=(1, 1), use_bias=False)(x35)  # ex_conv
x37 = layers.Activation("sigmoid")(x36)

x37a = layers.UpSampling2D((7, 7))(x37)

# 7X7 Stage
x38 = Bottleneck_JQ(512, 512)(x37a)
x39 = Bottleneck_JQ(512, 512)(x38)
x40 = Bottleneck_JQ(512, 512)(x39)
x41 = Bottleneck_JQ(512, 512)(x40)
x42 = Bottleneck_JQ(512, 512)(x41)
x43 = Bottleneck_JQ(512, 512)(x42)

# BODY
# 14x14 upsampling
x44 = layers.UpSampling2D((2, 2))(x43)

# 14X14 Stage
x45 = Bottleneck_JQ(512, 256)(x44)
x46 = Bottleneck_JQ(256, 256)(x45)
x46a = layers.Concatenate()([x44, x46])  # concatenate
# x47 = Bottleneck_JQ(256,384)(x48)
x47 = Bottleneck_JQ(768, 384, mode="UP", k=2)(x46a)
x48 = Bottleneck_JQ(384, 384)(x47)

x48a = layers.UpSampling2D((2, 2))(x48)

# 28X28 Stage
x49 = Bottleneck_JQ(384, 128)(x48a)
x50 = Bottleneck_JQ(128, 128)(x49)
x50a = layers.Concatenate()([x48a, x50])  # concatenate
x51 = Bottleneck_JQ(512, 256, mode="UP", k=2)(x50a)
# x51 = Bottleneck_JQ(128,256)(x50)
x52 = Bottleneck_JQ(256, 256)(x51)

x52a = layers.UpSampling2D((2, 2))(x52)

# 56X56 Stage
x53 = Bottleneck_JQ(256, 64)(x52a)
x54 = Bottleneck_JQ(64, 64)(x53)
x54a = layers.Concatenate()([x52a, x54])  # concatenate
x55 = Bottleneck_JQ(64, 320)(x54a)
x56 = Bottleneck_JQ(320, 320)(x55)

x56a = layers.MaxPool2D(2, strides=2)(x56)

# HEAD
# 28X28 Stage
x57 = Bottleneck_JQ(320, 512)(x56a)
x58 = Bottleneck_JQ(512, 512)(x57)
x58a = layers.Concatenate()([x56a, x58])  # concatenate
x59 = Bottleneck_JQ(512, 832)(x58a)
x60 = Bottleneck_JQ(832, 832)(x59)

x60a = layers.MaxPool2D(2, strides=2)(x60)

# 14X14 Stage
x61 = Bottleneck_JQ(832, 768)(x60a)
x62 = Bottleneck_JQ(768, 768)(x61)
# x63 = Bottleneck_JQ(768,1600)(x62)
x63a = layers.Concatenate()([x60a, x62])  # concatenate
x64 = Bottleneck_JQ(1600, 1600)(x63a)
x65 = Bottleneck_JQ(1600, 1600)(x64)
x66 = Bottleneck_JQ(1600, 1600)(x65)

x66a = layers.MaxPool2D(2, strides=2)(x66)

# 7X7 Stage
x67 = Bottleneck_JQ(1600, 512)(x66a)
x68 = Bottleneck_JQ(512, 512)(x67)
x69 = Bottleneck_JQ(512, 512)(x68)
x70 = Bottleneck_JQ(512, 512)(x69)

x71 = layers.Concatenate()([x66a, x70])

x72 = layers.BatchNormalization()(x71)
x73 = layers.Activation("relu")(x72)
x74 = layers.Conv2D(1056, 1, activation=None, padding="same", strides=(1, 1), use_bias=False)(x73)
x75 = layers.BatchNormalization()(x74)
x76 = layers.GlobalAveragePooling2D()(x75)  # does not translate from pytorch perfectly
x77 = layers.Reshape((1, 1, 1056))(x76)
x78 = layers.Conv2D(number_classes, 1, activation=None, padding="same", strides=(1, 1), use_bias=False)(x77)

# output layers (these aren't part of FishNet, this is our own top layer to make it fit the dataset we are using)
img_outputs2 = layers.Flatten()(x78)
test_mdl1 = keras.Model(img_inputs, img_outputs2, name="test_mdl1")
test_mdl1.summary()


def lr_scheduler(epoch, lr):
    if (epoch + 1) % 30 == 0:
        print("deacay learning rate")
        return lr * 0.1
    else:
        return lr


def lr_scheduler_w(epoch, lr, weight_decay):
    if (epoch + 1) % 30 == 0:
        print("deacay learning rate and decay weight decay")
        print(lr, weight_decay)
        return lr * 0.1, weight_decay * 0.1
    else:
        return lr, weight_decay


opt = tfa.optimizers.SGDW(weight_decay=0.0001, learning_rate=0.05, momentum=0.9)
# opt = keras.optimizers.SGD(learning_rate=0.05, momentum=0.9)  # TODO:weight decay
test_mdl1.compile(
    optimizer=opt,
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# data setup
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
train_path = "/root/subimagenet/train"
train_it = train_datagen.flow_from_directory(train_path, target_size=(224, 224), shuffle=True, batch_size=64)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_path = "/root/subimagenet/val"
val_it = val_datagen.flow_from_directory(val_path, target_size=(224, 224), shuffle=False, batch_size=64)

# fit
history = test_mdl1.fit(
    x=train_it,
    epochs=100,
    validation_data=val_it,
    workers=8,
    callbacks=[WandbCallback(), LearningRateScheduler(lr_scheduler_w)],
)
hist = pd.DataFrame(history.history)

fig, ax = plt.subplots(1, 2, figsize=(15, 4))
hist.plot(y=["loss", "val_loss"], ax=ax[0])
hist.plot(y=["accuracy", "val_accuracy"], ax=ax[1])
