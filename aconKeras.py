# An unofficial inplementation of ACON follows: https://github.com/nmaac/acon
# Created by Wang Dai on Oct. 11, 2021
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, multiply, add, Multiply
from tensorflow.keras.layers import Conv2D, BatchNormalization

class Acon(Layer):
    def __init__(self, inChannels, **kwargs):
        self.inChannels = inChannels
        super(Acon, self).__init__(**kwargs)

    def build(self, input_shape):
        self.p1 = self.add_weight(name='p1',
                                  shape=(1, 1, self.inChannels),
                                  initializer="random_normal",
                                  trainable=True)
        self.p2 = self.add_weight(name='p2',
                                  shape=(1, 1, self.inChannels),
                                  initializer="random_normal",
                                  trainable=True)
        oneInitializer = keras.initializers.Ones()
        self.beta = self.add_weight(name="beta",
                                    shape=(1, 1, self.inChannels),
                                    initializer=oneInitializer,
                                    trainable=True)
        # toFix:
        # Here when the self.beta is created as a tf.Variable, excute "multiply([self.beta, x])" will given such error
        # Error info:
        #   RuntimeError: Variable *= value not supported. Use `var.assign(var * value)` to modify the variable or `var = var * value` to get a new Tensor object.
        # It seems keras.layers.multiply(or add) can't use a tf.Variable as input?
        self.beta = tf.convert_to_tensor(self.beta)
        self.p2 = tf.convert_to_tensor(self.p2)

        super(Acon, self).build(input_shape)

    def call(self, x):
        z1 = multiply([self.p1 - self.p2, x])
        return add([multiply([z1, keras.layers.Activation("sigmoid")(multiply([self.beta, z1]))]), multiply([self.p2, x])])

class MetaAcon(Layer):
    def __init__(self, inChannels, reduction, **kwargs):
        self.inChannels = inChannels
        self.reduction = reduction
        super(MetaAcon, self).__init__(**kwargs)
    def build(self, input_shape):
        self.p1 = self.add_weight(name='p1',
                                  shape=(1, 1, self.inChannels),
                                  initializer="random_normal",
                                  trainable=True)
        self.p2 = self.add_weight(name='p2',
                                  shape=(1, 1, self.inChannels),
                                  initializer="random_normal",
                                  trainable=True)

        self.p2 = tf.convert_to_tensor(self.p2)

        mediaChannels = max(self.reduction, self.inChannels//self.reduction)
        self.conv_1 = Conv2D(mediaChannels, kernel_size=1, strides=(1,1), data_format="channels_last", use_bias=True)
        self.bn1 = BatchNormalization()
        self.conv_2 = Conv2D(self.inChannels, kernel_size=1, strides=(1,1), data_format="channels_last", use_bias=True)
        self.bn2 = BatchNormalization()
        self.nlin = keras.layers.Activation("sigmoid")
        super(MetaAcon, self).build(input_shape)

    def call(self, x):
        # toFix:    https://blog.csdn.net/baidu_38008726/article/details/116723041
        beta = K.mean(K.mean(x, axis=1, keepdims=True), axis=2, keepdims=True)
        beta = self.conv_1(beta)
        beta = self.bn1(beta)
        beta = self.conv_2(beta)
        beta = self.bn2(beta)
        beta = self.nlin(beta)

        z1 = multiply([self.p1 - self.p2, x])
        return add([multiply([z1, self.nlin(multiply([beta, z1]))]), multiply([self.p2, x])])



