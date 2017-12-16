import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Input
from keras.layers import GRU, Bidirectional
import keras.models as kmodels
from keras.utils.data_utils import get_file
import numpy as np
import random

sample_num = 12000
input_length = 100 #k

X_train = np.load("data/train_out_G_1.npy") 
X_test = np.load("data/test_out_G_5.npy")
Y = np.load("data/out.npy")

X_train = np.true_divide((X_train - np.ndarray.min(X_train)), (np.ndarray.max(X_train) - np.ndarray.min(X_train)))
X_test = np.true_divide((X_test - np.ndarray.min(X_test)), (np.ndarray.max(X_test) - np.ndarray.min(X_test)))

X_train = X_train.reshape((12000,100,2))
X_test = X_test.reshape((12000,100,2))
Y_test = Y.reshape((12000,1,100))

print('Build model...')
model = Sequential()
input = Input(shape=(100,2))
GRU_1 = Bidirectional(GRU(200, return_sequences=True))(input)
norm_1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, input_shape=(100,400))(GRU_1)
print(input._keras_shape, GRU_1._keras_shape, norm_1._keras_shape)
GRU_2 = Bidirectional(GRU(200))(norm_1)
norm_2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, input_shape=(100,400))(GRU_2)
out = Dense(100, activation='sigmoid')(norm_2)
print model.summary()

model = kmodels.Model(input, out)
adam = keras.optimizers.Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy', 'mse'])
model.fit(X_train, Y, nb_epoch=100,batch_size=200, validation_data=(X_test, Y))

predict = model.predict(X_test)
