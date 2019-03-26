import csv
import sys

import matplotlib.pyplot as plt

results = []

reader = csv.reader(sys.stdin)
next(reader)
for row in reader:
    try:
        *_, result = map(float, row)
    except ValueError as e:
        print(row)
        raise e
    results.append(result)

plt.plot(results)
plt.xlabel('hours')
plt.ylabel('temperature')
plt.show()
