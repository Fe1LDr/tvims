import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels
import statsmodels.stats.weightstats as w
from scipy.stats import t

if __name__ == '__main__':
    # берем данные из файла
    DataFrame = pd.read_excel('data1.xlsx', index_col=None)
    # из созданного фрейма берем значения
    Numbers = DataFrame['значение'].tolist()
    # сортировка для вариационного ряда
    Sorted = np.sort(Numbers)
    mean = np.mean(Sorted)
    print('выборочное среднее: ', mean)
    print('выборочная дисперсия: ', np.var(Sorted, ddof=0))
    print('исправленная выборочная дисперсия: ', np.var(Sorted, ddof=1))
    var = np.var(Sorted, ddof=1)
    # доверительный интервал для мат ожидания
    alpha = 0.05
    print('доверительный интервал для мат ожидания')
    print(w.zconfint(Sorted, alpha=0.05, alternative="two-sided"))
    # доверительный интервал для дисперсии
    chi_l, chi_r = stats.chi2.ppf([1-alpha/2, alpha/2], df=49)
    left = 49 * var / chi_l
    right = 49 * var / chi_r
    print('доверительный интервал для дисперсии')
    print(left, right)
    # доверительный интервал для мат ожидания (группированная выборка)
    t1 = 2.0096
    x = 6.1
    S = 69.4286
    leftgr = x - t1 * S**(1/2) / (50)**(1/2)
    rightgr = x + t1 * S**(1/2) / (50)**(1/2)
    print('доверительный интервал для мат ожидания')
    print(leftgr, rightgr)
    # доверительный интервал для дисперсии (группированная выборка)
    left = 49 * S / chi_l
    right = 49 * S / chi_r
    print('доверительный интервал для дисперсии')
    print(left, right)
    # print(stats.chi2.ppf(0.05, df=49))
    print(var**0.5)
    print(mean+0.5*var**0.5)
    print('расчет значимости гипотезы')
    print(stats.ttest_1samp(Sorted, mean+0.5*var**0.5, axis=0, nan_policy='propagate'))

    UniqueNumbers, inverse = np.unique(Numbers, return_inverse=True)
    NumberOfAppearancesData = np.bincount(inverse)
    print(stats.chisquare(NumberOfAppearancesData, f_exp=None, ddof=0, axis=0))

    intervalsPol = [-10, -5, 0, 5, 10, 15, 20, 25]
    n, counts, bins = plt.hist(Sorted, bins=intervalsPol, histtype='step')
    print(n)
    print('реализация критерия хи-квадрат')
    print(stats.chisquare(n, f_exp=None, ddof=0, axis=0))
    print(stats.chi2.ppf(0.995, df=49))

