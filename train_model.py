from model import get_model
import numpy as np
import argparse

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

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mse'])
model.fit(trainX, trainY, epochs=50, batch_size=32 )