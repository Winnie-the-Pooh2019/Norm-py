from math import sqrt
import numpy as np
from scipy.stats import norm


def min_neighbour_index(freq, index):
    if index == 0:
        return index + 1
    elif index == len(freq) - 1:
        return index - 1
    elif freq[index + 1] > freq[index - 1]:
        return index - 1
    else:
        return index + 1


class Interval:
    def __int__(self, values, bounds, frequency):
        self.values = values
        self.bounds = bounds
        self.frequency = frequency
        th_frequency = 0.0


n = 100
mx = 20
sx = 2

r = np.random.uniform(size=(n, 12))
z = np.sum(r, axis=1) - 6
x = mx + z * sx

mean = np.mean(x)
std = np.std(x)

mean_err = abs(mean - mx)
std_err = abs(std - sx)

print(f'Мат. ожидание: {mean}')
print(f'Стандартное отклонение: {std}')
print(f'Ошибки оценивания мат. ожидания: {mean_err}')
print(f'Ошибки оценивания стандартного отклонения: {std_err}')

k = 1 + 3.3221 * np.log10(n)
h = (max(x) - min(x)) / k
iss = np.arange(min(x), max(x), h)  # разбиение интервалов
intervals = [(iss[i], iss[i + 1]) for i in range(len(iss) - 1)]
xi = [it[0] + it[1] / 2 for it in intervals]
ui = (xi - mean) / std
elements_in_interval = [[e for e in x if interval[0] <= e <= interval[1]] for interval in intervals]
frequency = [len(e) for e in elements_in_interval]
fqs = n * h / std * norm.pdf(ui)
freq_theoretical = [f for f in fqs]

print("h = ", h)
print("freqs = ", frequency)
# print(elements_in_interval)

i = len(intervals) - 1
while i in range(len(intervals)):
    if frequency[i] < 5:
        target = min_neighbour_index(frequency, i)

        freq_theoretical[i] += freq_theoretical[target]
        del freq_theoretical[target]

        intervals[i] = (min(intervals[i][0], intervals[i][1], intervals[target][0], intervals[target][1]),
                        max(intervals[i][0], intervals[i][1], intervals[target][0], intervals[target][1]))
        del intervals[target]

        elements_in_interval[i] += elements_in_interval[target]
        del elements_in_interval[target]

        frequency[i] += frequency[target]
        del frequency[target]

    i -= 1

print(f'freqs = {frequency}')

# for interval in intervals:
#     elements_in_interval.append(0)
#
#     for e in x:
#         if ()
