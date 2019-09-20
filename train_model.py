from model import get_model
import numpy as np
import argparse
import keras


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainX', help='trainX npy', type=str, required=True)
    parser.add_argument('--trainY', help='trainY npy', type=str, required=True)
    parser.add_argument('--testX', help='testX npy', type=str, required=True)
    parser.add_argument('--testY', help='testY npy', type=str, required=True)
    args = parser.parse_args()
    return args

args = get_args()

print("loading data")
trainX, trainY = np.load(args.trainX), np.load(args.trainY)
testX, testY = np.load(args.testX), np.load(args.testY)
print("done loading data.")

model = get_model(trainX.shape[1])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss='mse',
              metrics=['mse'])

chck = keras.callbacks.callbacks.ModelCheckpoint("checkpoints", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

model.fit(trainX, trainY, validation_data=[testX, testY], epochs=200, batch_size=128, verbose=1, callbacks=[chck])