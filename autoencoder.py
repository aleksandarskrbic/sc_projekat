import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np 
from tensorflow import set_random_seed
import os

class AutoEncoder:
    def __init__(self, input_data):
        self.encoding_dim = 1000
        self.x = input_data

    def _encoder(self):
        inputs = Input(shape=(self.x[0].shape))
        encoded_layer_1 = Dense(8000, activation='relu')(inputs)
        encoded_layer_2 = Dense(3000, activation='relu')(encoded_layer_1)
        encoded = Dense(self.encoding_dim, activation='relu')(encoded_layer_2)
        model = Model(inputs, encoded)
        self.encoder = model
        return model

    def _decoder(self):
        inputs = Input(shape=(self.encoding_dim,))
        decoder_layer_1 = Dense(3000, activation='relu')(inputs)
        decoder_layer_2 = Dense(8000, activation='relu')(decoder_layer_1)
        decoded = Dense(self.x.shape[1])(decoder_layer_2)
        model = Model(inputs, decoded)
        self.decoder = model
        return model

    def encoder_decoder(self):
        ec = self._encoder()
        dc = self._decoder()
        
        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = Model(inputs, dc_out)
        
        self.model = model
        return model

    def fit(self, batch_size=20, epochs=80):
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(self.x, self.x,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10),
                                   ModelCheckpoint(filepath='best_ae.h5', monitor='val_loss', save_best_only=True)])

    def save(self):
        self.encoder.save('encoder_weights.h5')
        self.decoder.save('decoder_weights.h5')
        self.model.save('ae_weights.h5')
