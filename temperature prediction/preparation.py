import argparse
import csv
import sys
from calendar import isleap
from datetime import datetime
from itertools import chain
from operator import itemgetter

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--city", nargs="?", type=str, default="Vancouver")
ap.add_argument("-b", "--bound", nargs="?", type=int, default=30)
ap.add_argument("-w", "--window", nargs="?", type=int, default=24*7)
args = ap.parse_args()

city = args.city
bound = args.bound
double_bound = bound * 2
window_len = args.window


# datetime -> hour of year -> [0, 1]
def norm_date(date: str) -> float:
    dt = datetime.strptime(date, "%d/%m/%Y %H:%M")
    dtt = dt.timetuple()
    days_in_year = 366 if isleap(dt.year) else 365
    hours_in_year = days_in_year * 24
    return (dtt.tm_yday * 24 + dtt.tm_hour) / hours_in_year


# temperature -> [-30 C, +30 C] -> [0, 1]
def norm_temperature(kelvin: float) -> float:
    celsius = kelvin - 273.16
    norm = celsius + bound
    if norm < 0:
        norm = 0
    elif norm > double_bound:
        norm = double_bound
    return norm / double_bound


data = []
previous = None
r = csv.DictReader(sys.stdin)
for row in r:
    if not row[city]:
        if previous:
            temperature = previous
        else:
            continue
    else:
        temperature = norm_temperature(float(row[city]))
        previous = temperature
    data.append((norm_date(row["Date"]), temperature))

window = data[:window_len]
print(*map(lambda x: f"prev-{x}", range(window_len, 0, -1)), "Date", "Temperature", sep=",")
for i, row in enumerate(data):
    if i < window_len:
        continue
    print(*chain(map(itemgetter(1), window), row), sep=",")
    window = data[i - window_len + 1:i + 1]
