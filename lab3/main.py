import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as seaborn

if __name__ == '__main__':
    # берем данные из файла
    DataFrame = pd.read_excel('data2.xlsx', index_col=None)
    # из созданного фрейма берем значения
    X = DataFrame['X'].tolist()
    Y = DataFrame['Y'].tolist()
    SortedX = np.sort(X)
    SortedY = np.sort(Y)
    plt.scatter(X, Y)
    plt.xticks([5, 8, 11, 14, 17])
    plt.yticks([9, 15, 21, 27, 33, 39])
    plt.grid()
    x1, y1 = [5, 16], [13.17, 34.9]
    x2, y2 = [5.83, 15.51], [9, 39]
    plt.plot(x1, y1, x2, y2)
    plt.show()

    seaborn.pairplot(DataFrame, kind='reg', diag_kind='auto', height=4)
    #plt.show()

    print('Ковариационная матрица')
    print(np.cov(X, Y))
    print('Корреляционная матрица')
    print(np.corrcoef(X, Y))

    seaborn.heatmap(DataFrame.corr(), annot=True, fmt='.2g', linewidth=0, linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None)
    #plt.show()

    print('Выборочные характеристки для X: ')
    print('Выборочное среднее: ', np.mean(X))
    print('Выборочная дисперсия: ', np.var(X))
    print('Исправленная выборочная дисперсия: ', np.var(X, ddof=1))
    print('------------')
    print('Выборочные характеристки для Y: ')
    print('Выборочное среднее: ', np.mean(Y))
    print('Выборочная дисперсия: ', np.var(Y))
    print('Исправленная выборочная дисперсия: ', np.var(Y, ddof=1))
    print('------------')
    print('Группированные характеристики: ')
    n, counts, bins = plt.hist(SortedX, bins=[5, 8, 11, 14, 17], histtype='step')
    midpoints1 = 0.5 * (counts[1:] + counts[:-1])
    result1 = 0
    for i in range(len(midpoints1)):
        result1 += n[i]/50 * midpoints1[i]
    print('Выборочное среднее для X', result1)
    mat1 = result1

    result1 = 0
    for i in range(len(midpoints1)):
        result1 += (n[i]*(midpoints1[i])**2)/50
    print('Выборочная дисперсия для X: ', result1-mat1**2)
    print('Исправленная выборочная дисперсия для X: ', 50/49*(result1 - mat1 ** 2))

    n, counts, bins = plt.hist(SortedY, bins=[9, 15, 21, 27, 33, 39], histtype='step')
    midpoints2 = 0.5 * (counts[1:] + counts[:-1])
    result2 = 0
    for i in range(len(midpoints2)):
        result2 += n[i] / 50 * midpoints2[i]
    print('Выборочное среднее для Y', result2)
    mat2 = result2

    result2 = 0
    for i in range(len(midpoints2)):
        result2 += (n[i]*(midpoints2[i])**2)/50
    print('Выборочная дисперсия для Y: ', result2-mat2**2)
    print('Исправленная выборочная дисперсия для Y: ', 50 / 49 * (result2 - mat2 ** 2))

    print('Сумма квадратов, обусловленная регрессией Q_R')
    Qr = 0
    Qr = 49 * 10.5135**2/5.2924
    print(Qr)
    print('Q_y')
    Qy = 0
    y = np.mean(SortedY)
    for i in range(len(SortedY)):
        Qy += (SortedY[i]-y)**2
    print(Qy)
    print('Q_e')
    Qe = 0
    Qe = Qy - Qr
    print(Qe)
    sum = 0
    for i in range(len(SortedX)):
        sum += SortedX[i]
    print(sum)


