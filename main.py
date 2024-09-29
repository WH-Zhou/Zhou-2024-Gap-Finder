import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from typing import List, Tuple
from functools import lru_cache
import os
import imageio.v2 as imageio
from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',
    "font.size": 20,
    "mathtext.fontset":'stix',
    # "font.serif": ['SimSun'],
}
rcParams.update(config)

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
} 



# @lru_cache
def read_data(filename: str = None):
    merge_df = pd.read_csv(filename)
    x, y = merge_df['diameter (km)'], merge_df['Period (h)']
    x, y = np.log10(x), -np.log10(y)
    X = np.array(list(zip(x, y)))
    # plt.plot(x,y,'.')
    return X


def line(X, k, b, gap=0.5):
    return k * X[:, 0] + b + gap


def classify(X, k, b, gap) -> List[int]:
    upper_index = X[:, 1] > line(X, k=k, b=b, gap=gap)  # label 1
    lower_index = X[:, 1] < line(X, k=k, b=b, gap=-gap)   # label 2
    #  [0,1,1,1,0]
    labels = [-1] * len(X)
    for i in range(len(X)):
        if upper_index[i]:
            labels[i] = 1
        elif lower_index[i]:
            labels[i] = 0
    labels = np.array(labels)
    return labels


def split(X, labels):
    ratio = 1
    with_labels = labels != -1
    lower = labels == 0
    upper = labels == 1
    lower_count = sum(lower)
    upper_count = sum(upper)

    X_upper_train, y_upper_train = X[upper], labels[upper]
    X_lower_train, y_lower_train = X[lower], labels[lower]

    random_index = np.arange(int(upper_count))
    np.random.shuffle(random_index)
    X_upper_train, y_upper_train = X_upper_train[random_index], y_upper_train[random_index]
    upper_count = int(ratio * upper_count)
    X_upper_train, y_upper_train = X_upper_train[:upper_count, :], y_upper_train[:upper_count]
    random_index = np.arange(lower_count)
    np.random.shuffle(random_index)
    X_lower_train, y_lower_train = X_lower_train[random_index], y_lower_train[random_index]

    X_test, y_test = X[~with_labels], labels[~with_labels]
    X_train = np.r_[X_upper_train, X_lower_train]
    y_train = np.r_[y_upper_train, y_lower_train]
    # print(f'upper ~ lower: {upper_count}~{lower_count}')
    return (X_train, y_train), (X_test, y_test)


def fit(X, labels) -> Tuple[float, float]:
    clf = svm.LinearSVC(dual='auto')
    fit = clf.fit(X, labels)
    coefs = fit.coef_[0]
    intercept = fit.intercept_[0]
    k = -coefs[0]/coefs[1]
    b = -intercept/coefs[1]
    return k, b, clf


def fill_gap(k, b, gap, color, savefig_name=None):
    x = np.linspace(0, 2.5, 100)
    y = k*x + b
    lower = y-gap/2
    upper = y+gap/2
    plt.plot(x, y+gap/2, color=color)
    plt.plot(x, y-gap/2, color=color)
    plt.fill_between(x, lower, upper, color='grey', alpha=0.5)
    if savefig_name:
        plt.savefig(savefig_name)


def plot_original_data():
    filename = "asteroid_dataframe.csv"   # the original data file name (csv format)
    X = read_data(filename)
    
    plt.figure(figsize=(10, 8))
    plt.xlim([0, 3])
    plt.ylim([3.5, 0.2])

    # plot the gap
    x = np.linspace(0, 3.1, 100)
    k = 0.6
    b = 1.2
    gap = 0.4
    y = k*x + b
    lower = y-gap/2
    upper = y+gap/2
    plt.plot(x, y+gap/2, color= 'k')
    plt.plot(x, y-gap/2, color= 'k')
    plt.fill_between(x, lower, upper, color='grey', alpha=0.5)

    # plot the original data
    condition1 =  ( 0.6 * X[:, 0] + 1.2 - 0.2 >  -X[:, 1]) 
    condition2 =  (0.6 * X[:, 0] + 1.2 + 0.2 < - X[:, 1])
    X1 = X[condition1]
    X2 = X[condition2]
    X3 = X[~(condition1 | condition2)]

    plt.scatter(X1[:, 0],- X1[:, 1], s=1.0, label='Class I')
    plt.scatter(X2[:, 0],- X2[:, 1], s=1.0, label = 'Class II')
    plt.scatter(X3[:, 0],- X3[:, 1], color = 'grey', s=1.0, label = 'Unknown')

    plt.legend(fontsize = 15)
    plt.xlabel('log(D) (km)', font = font1)
    plt.ylabel('log(P) (h)', font = font1)
    plt.title('Initialization:  $P_{\\rm h} = %.2f \\, D_{\\rm km}^{%.3f} $'%(10**(1.2),-0.6), fontdict=font1)
    # plt.gca().invert_yaxis()

    plt.savefig("initialization.pdf", bbox_inches='tight')
    plt.show()

def generate_gif(n_image):
    images = []
    for i in range(n_image):
        images.append(imageio.imread("gif/gap{}.png".format(i)))
        os.remove("gif/gap{}.png".format(i))
    imageio.mimsave('gif/gap_identification.gif', images, duration=0.1)


def fit_params(savefig_filename=None):
    # 读取数据
    filename = "asteroid_dataframe.csv"   # the original data file name (csv format)
    X = read_data(filename)
    k, b, gap = -0.4, -1.2, 0.2
    clf = None
    plt.figure(figsize=(10, 8))
    for i in range(150):
        labels = classify(X, k=k, b=b, gap=gap)
        (X_train, y_train), (X_test, y_test) = split(X, labels)

        # 获取直线斜率
        k, b, clf = fit(X_train, y_train)
        print(f'Iteration: {i}, k: {k}, b: {b}')
        
        # plot
        plt.clf()
        y_pred = clf.predict(X)
        X1 = X[y_pred == 1]
        X2 = X[y_pred == 0]
        plt.scatter(X1[:, 0], -X1[:, 1], s=1.0, label='Class I')
        plt.scatter(X2[:, 0], -X2[:, 1], s=1.0, label = 'Class II')

        x_min, x_max = 0, 3
        x = np.linspace(x_min, x_max, 100)
        plt.plot(x, - (k * x + b), color = 'black')
        plt.plot(x, - (k * x + b+gap))
        plt.plot(x, - (k * x + b-gap))
        plt.fill_between(x, - (k * x + b-gap), - (k * x + b+gap), color='grey', alpha=0.5)
        plt.gca().invert_yaxis()
        plt.legend(fontsize = 15)
        plt.xlabel('log(D) (km)', font = font1)
        plt.ylabel('log(P) (h)', font = font1)
        plt.title('Iteration %d:  $P_{\\rm h} = %.2f \\, D_{\\rm km}^{%.3f} $'%(i+1, 10**(-b),k), fontdict=font1)
        plt.xticks(fontproperties = 'Times New Roman', size = 20)
        plt.yticks(fontproperties = 'Times New Roman', size = 20)   
        plt.xlim([0, 3])
        plt.ylim([3.5, 0.2])
        plt.pause(0.1)
        # if i == 0 or i == 149:
        #     plt.savefig("gif/gap{}.pdf".format(i), bbox_inches='tight')
        # plt.savefig("gif/gap{}.png".format(i))
    
    # generate_gif(150)
        
    if savefig_filename:
        plt.savefig(savefig_filename)
    plt.show()
    


if __name__ == '__main__':
    fit_params()
    # plot_original_data()
    # imageio.imread("gif/gap{}.pdf".format(2))