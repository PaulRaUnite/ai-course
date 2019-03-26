import argparse
import csv
import sys

import numpy as np
from keras.engine.saving import load_model

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="network.model")
ap.add_argument("-b", "--bound", default=30)

args = ap.parse_args()
bound = args.bound
filename = args.model


def denormalizing(norm: float) -> float:
    return norm * bound * 2 - bound


# load the model and label binarizer
print("[INFO] loading network...")
model = load_model(filename)

reader = csv.reader(sys.stdin)
next(reader)
data = []
results = []
for row in reader:
    *input_data, result = map(float, row)
    data.append(input_data)
    results.append(result)

predicted = model.predict(np.array(data))
for i in range(len(predicted)):
    print(f"Predicted: {denormalizing(predicted[i][0])}, actual: {denormalizing(results[i])}.")
