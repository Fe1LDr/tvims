import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels
from statsmodels.distributions.empirical_distribution import ECDF
from tabulate import tabulate

if __name__ == '__main__':
    # берем данные из файла
    DataFrame = pd.read_excel('data1.xlsx', index_col=None)
    print(DataFrame)
    # из созданного фрейма берем значения
    Numbers = DataFrame['значение'].tolist()
    print(Numbers)
    # списки уникальных чисел и их количество
    # для формирования вариационного и статистического рядов
    UniqueNumbers, inverse = np.unique(Numbers, return_inverse=True)
    print(UniqueNumbers)
    NumberOfAppearancesData = np.bincount(inverse)
    print(NumberOfAppearancesData)
    # сортировка для вариационного ряда
    Sorted = np.sort(Numbers)
    print(Sorted)
    intervalsPol = [-10, -5, 0, 5, 10, 15, 20, 25]
    intervalsHist = [-10, -5, 0, 5, 10, 15, 20, 25]
    # вывод статистического ряда
    print(tabulate([NumberOfAppearancesData, NumberOfAppearancesData / 50], headers=UniqueNumbers))
    # построение гистограммы относительных частот / 7 интервалов
    # n, bins, patches = plt.hist(Sorted, bins=7, density=True, weights=NumberOfAppearancesData, cumulative=False,
    #                        histtype='bar', log=False)
    # plt.show()
    # построение гистограммы частот / 7 интервалов
    plt.xticks(intervalsPol)
    n, counts, bins = plt.hist(Sorted, bins=intervalsPol, histtype='step')
    midpoints = 0.5*(counts[1:]+counts[:-1])
    plt.plot(midpoints, n)
    plt.show()
    y = ECDF(Sorted)
    plt.step(y.x, y.y)
    plt.xlabel('$x$')
    plt.ylabel('$F(x)$')
    # y = y(Sorted)
    # print(y)
    # plt.plot(Sorted, y)
    # plt.show()
    plt.hist(Sorted, bins=intervalsHist, density=True, weights=NumberOfAppearancesData, cumulative=True,
             histtype='step', color='red')
    plt.show()
    # размах выборки
    print(Sorted[49] - Sorted[0])
    # мат ожидание
    print(np.mean(Sorted))
    # дисперсия
    print(np.var(Sorted))
    # медиана
    print(np.median(Sorted))
    # группированный размах
    print(max(counts) - min(counts))
    # группированное мат ожидание
    result = 0
    for i in range(len(midpoints)):
        result += n[i]/50 * midpoints[i]
    print(result)
    mat = result
    # группированная дисперсия
    result = 0
    print(n)
    print(midpoints)
    for i in range(len(midpoints)):
        result += (n[i]*(midpoints[i])**2)/50
    print(result-mat**2)
    # группированная медиана
    print(np.median(midpoints))
