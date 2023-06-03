from math import sqrt
import numpy as np
from scipy.stats import norm, chi2, chisquare


def min_neighbour_index(freq, index):
    if index == 0:
        return index + 1
    elif index == len(freq) - 1:
        return index - 1
    elif freq[index + 1] > freq[index - 1]:
        return index - 1
    else:
        return index + 1


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
iss = np.arange(min(x), max(x) + h, h)  # разбиение интервалов
xi = (iss[:-1] + iss[1:]) / 2
ui = (xi - mean) / std
fqs = n * h / std * norm.pdf(ui)

intervals = [(iss[i], iss[i + 1]) for i in range(len(iss) - 1)]
elements_in_interval = [[e for e in x if interval[0] <= e <= interval[1]] for interval in intervals]
frequency = [len(e) for e in elements_in_interval]
print("FREQS", sum(frequency))
freq_theoretical = [f for f in fqs]

print("h = ", h)
print("freqs = ", frequency)

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
print(f'freqs_th = {freq_theoretical}')

hi = 0
for i in range(len(intervals)):
    hi += (((frequency[i] - freq_theoretical[i]) ** 2) / freq_theoretical[i])

hi_list = [3.8, 6.0, 7.8, 9.5, 11.1, 12.6, 14.1, 15.5]
hi_crit = hi_list[len(frequency) - 3]

print(f'Число степеней свободы = {len(frequency) - 3}')
print(f'Х2 наблюдаемый = {hi}')
print(f'Х2 критический = {hi_crit}')

if hi > hi_crit:
    print("Гипотеза отвергается")
else:
    print("Гипотеза принимается")

lamda_crit = 1.36

# intervals = [(iss[i], iss[i + 1]) for i in range(len(iss) - 1)]
# elements_in_interval = [[e for e in x if interval[0] <= e <= interval[1]] for interval in intervals]
# frequency = [len(e) for e in elements_in_interval]

freq_acc = [sum(frequency[:i + 1]) for i in range(len(frequency))]
xi = np.array([(interval[0] + interval[1]) / 2 for interval in intervals])
dist = 0.5 + norm.cdf((xi - mx) / sx)
dist_n = np.array([fa / n for fa in freq_acc])
dist_diff = np.abs(dist - dist_n)
max_diff = max(dist_diff)
lamda = max_diff * sqrt(n)

print(lamda)

if lamda > lamda_crit:
    print("Гипотеза отвергается")
else:
    print("Гипотеза подтверждается")

print(f'dist = {dist}')
print(f'dist_n = {dist_n}')
print(f'dist_diff = {dist_diff}')
