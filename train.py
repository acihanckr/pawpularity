from tensorflow.python.keras import utils
from data import Data
import tensorflow.keras as keras
import layers
import tensorflow as tf
from utils import train_alive
from threading import Thread

BASE_DIR = '.'
TRAIN_FILE = BASE_DIR + 'train.csv'
TRAIN_FOLDER = BASE_DIR + '/train/'

TEST_FILE = BASE_DIR + '/test.csv'
TEST_FOLDER = BASE_DIR + '/test/'

shape_to_cnn = (224,224,3)
scale = 1/127.5

#prepare tensorflow data
data = Data(base = BASE_DIR, train_file = TRAIN_FILE, \
            train_folder = TRAIN_FOLDER, test_file = TEST_FILE, \
            test_folder = TEST_FILE)
train_tfd, val_tfd, test_tfd = data.build_dataset()

#epochs and fitting parameters for both models are shared
epochs = 10
fitting_params = {'x':train_tfd, 'epochs':epochs,
                    'batch_size':64,
                    'validation_data':val_tfd}

#same inputs are shared for both models
inputs = keras.Input(shape = (None,None,3))

#build the first model
model1 = layers.get_input_layers(inputs, shape_out = shape_to_cnn,
                                    scale = scale)
model1 = layers.get_xception_layers(model1, shape_in = shape_to_cnn)
outputs1 = layers.get_dnn_layers(model1)

model1 = keras.Model(inputs, outputs1)
#print the first model
model1.summary()

#the callback for the first model
tb_callback1 = tf.keras.callbacks.TensorBoard(log_dir='logs/model1', update_freq = 100)

#compile the first model
model1.compile(optimizer = keras.optimizers.Adam(learning_rate = 3e-4),
              loss = keras.losses.MeanSquaredError(),
              metrics = [keras.metrics.RootMeanSquaredError(name="rmse")])

#build the second model
model2 = layers.get_input_layers(inputs, shape_out = shape_to_cnn,
                                    scale = scale)
model2 = layers.get_efficient_layers(model2)
outputs2 = layers.get_dnn_layers(model2)

model2 = keras.Model(inputs, outputs2)
#print the second model
model2.summary()

#the callback for the second model
tb_callback2 = tf.keras.callbacks.TensorBoard(log_dir='logs/model2', update_freq = 100)

#compile the second model
model2.compile(optimizer = keras.optimizers.Adam(learning_rate = 3e-4),
              loss = keras.losses.MeanSquaredError(),
              metrics = [keras.metrics.RootMeanSquaredError(name="rmse")])

#history objects are created
history1 = None
history2 = None

thread1 = Thread(target = train_alive, kwargs={'model':model1, 'callbacks' : [tb_callback1],
                    'fitting_params':fitting_params,'history':history1})
thread2 = Thread(target = train_alive, kwargs={'model':model2, 'callbacks' : [tb_callback2],
                    'fitting_params':fitting_params, 'history':history2})

#both models are trained simultaneously in different threads
thread1.start()
thread2.start()