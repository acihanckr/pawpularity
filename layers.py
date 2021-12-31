import tensorflow.keras as keras
from tensorflow.keras.applications import EfficientNetB3, Xception


def get_input_layers(model_in, shape_out, scale):
    x = keras.layers.Resizing(*shape_out[:2])(model_in)
    x = keras.layers.Rescaling(scale = scale, offset=-1)(x)
    return x

def get_xception_layers(model_in, shape_in, trainiable = False):
    x = keras.applications.Xception(
                weights = "imagenet", 
                input_shape = shape_in,
                include_top = False)
    x.trainable = trainiable
    x = x(model_in)
    return x

def get_efficient_layers(model_in):
    eff = EfficientNetB3(include_top=False)
    eff.trainable = False
    eff = eff(model_in)
    return eff
    
def get_dnn_layers(model_in):
    x = keras.layers.GlobalAveragePooling2D()(model_in)
    outputs = keras.layers.Dense(1, activation=None)(x)
    return outputs



