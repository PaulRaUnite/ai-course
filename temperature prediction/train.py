import argparse
import csv
import sys

import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epoches", nargs="?", type=int, default=30)
ap.add_argument("-r", "--rate", nargs="?", type=float, default=0.01)
ap.add_argument("-m", "--model", nargs="?", type=str, default="network.model")

args = ap.parse_args()
epochs = args.epoches
learning_rate = args.rate
filename = args.model

inputs = []
results = []

reader = csv.reader(sys.stdin)
next(reader)
for row in reader:
    try:
        *input_data, result = map(float, row)
    except ValueError as e:
        print(row)
        raise e
    inputs.append(input_data)
    results.append(result)

input_data = np.array(inputs, dtype="float")
results = np.array(results)

(trainX, testX, trainY, testY) = train_test_split(input_data, results, test_size=0.25, random_state=42)
window = input_data.shape[1]
model = Sequential()
model.add(Dense(365*24, input_shape=(window,), kernel_initializer='normal', activation="relu"))
model.add(Dense(365, kernel_initializer='normal', activation="relu"))
model.add(Dense(12*4, kernel_initializer='normal', activation="relu"))
model.add(Dense(12, kernel_initializer='normal', activation="relu"))
model.add(Dense(1, kernel_initializer='normal'))


print("[INFO] training network...")
opt = SGD(lr=learning_rate)
model.compile(loss="mean_squared_error", optimizer='adam',
              metrics=["mse", "mae", "mape", "cosine"])
# train the neural network
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              epochs=epochs, batch_size=32)

# save the model and label binarizer to disk
print("[INFO] serializing network...")
model.save(filename)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print("Mean squared log error:", mean_squared_log_error(testY, predictions[..., 0]))
