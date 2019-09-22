from model import get_model
import numpy as np
import tensorflow.keras as keras

from modelconfig import args


print("loading data")
data = np.load(args['cleaned_file'])
X_train, X_test = data['x_train'], data['x_test']
Y_train, Y_test = data['y_train'], data['y_test']
print("done loading data.")

model = get_model(args['vocab_size'], args['emb_size'], args['max_length'], args['h_size'])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss='mse',
              metrics=['mse'])

chck = keras.callbacks.ModelCheckpoint("checkpoints", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

model.fit(X_train, Y_train, validation_data=[X_test, Y_test], epochs=500, batch_size=128, verbose=1, callbacks=[chck])