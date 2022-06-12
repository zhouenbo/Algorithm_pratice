#! /usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Enbo Zhou"


import math
import matplotlib.pyplot as plt
import numpy as np

def main():
    f1 = lambda x: x / 3 - 5
    f2 = lambda x: -2 * pow(x, 2) + 7 * x
    f3 = lambda x: math.sqrt(x) - 2
    f4 = lambda x: pow(x, -2) - 3 * x
    f5 = lambda x: 2 * pow(x, 3) + 3 * pow(x, 2) - 12 * x + 4

    plt.figure(111)
    plt.subplot(321)
    t1 = np.arange(-10, 10, 0.1)
    y1 = [f1(x) for x in t1]
    plt.plot(t1, y1, '-r')
    plt.ylabel('x', fontsize=5)
    plt.ylabel('y', fontsize=5)
    plt.legend(labels=['y=x/3-5'], loc='best', fontsize=6)
    plt.xticks(range(-10, 11, 2), fontsize=5)
    plt.yticks(range(-10, 0, 2), fontsize=5)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.subplot(322)
    t2 = np.arange(-10, 10, 0.1)
    y2 = [f2(x) for x in t2]
    plt.plot(t2, y2, '-g')
    plt.ylabel('x', fontsize=5)
    plt.ylabel('y', fontsize=5)
    plt.legend(labels=['y=-2x^2+7x'], loc='best', fontsize=6)
    plt.xticks(range(-10, 11, 2), fontsize=5)
    plt.yticks(range(-250, 0, 50), fontsize=5)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.subplot(323)
    t3 = np.arange(0, 10, 0.1)
    y3 = [f3(x) for x in t3]
    plt.plot(t3, y3, '-b')
    plt.ylabel('x', fontsize=5)
    plt.ylabel('y', fontsize=5)
    plt.legend(labels=['y=sqrt(x)-2'], loc='best', fontsize=6)
    plt.xticks(range(0, 11, 2), fontsize=5)
    plt.yticks(range(-2, 3, 2), fontsize=5)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.subplot(324)
    t4_1 = np.arange(-10, -0.1, 0.05)
    t4_2 = np.arange(0.11, 10, 0.05)
    y4_1 = [f4(x) for x in t4_1]
    y4_2 = [f4(x) for x in t4_2]
    plt.plot(t4_1, y4_1, '-m', t4_2, y4_2, '-m')
    plt.ylabel('x', fontsize=5)
    plt.ylabel('y', fontsize=5)
    plt.legend(labels=['y=x^-2-3x'], loc='best', fontsize=6)
    plt.xticks(range(-10, 11, 2), fontsize=5)
    plt.yticks(range(-30, 101, 10), fontsize=5)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.subplot(325)
    t5 = np.arange(-10, 10, 0.1)
    y5 = [f5(x) for x in t5]
    plt.plot(t5, y5, '-k')
    plt.ylabel('x', fontsize=5)
    plt.ylabel('y', fontsize=5)
    plt.legend(labels=['y=2x^3+3x^2-12x+4'], loc='best', fontsize=6)
    plt.xticks(range(-10, 11, 2), fontsize=5)
    plt.yticks(range(-1600, 2200, 400), fontsize=5)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.show()


def plot_func():
    x = list(range(0,101,1))
    y1 = list(map(lambda s: s, x))
    y2 = list(map(lambda s: s*s, x))
    plt.figure()
    plt.plot(x,y1, label='Y = X')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(x,y2, label='Y = X^2')
    plt.legend()
    plt.show()

def plot_2():
    x = np.linspace(0,10,100)
    y = 25*x+750
    plt.figure()
    plt.plot(x,y,label='Y=25X+750')
    plt.ylabel('X/miles')
    plt.ylabel('Y/miles')
    plt.legend()
    plt.show()

def calculate():
    x_people = [1135.041721, 1136.436438, 1136.815817, 1139.819625]
    y_people = [375.625886, 374.766378, 374.294461, 376.3791]
    x_coffee = [1136.773759, 1135.817693, 1135.717992, 1135.982293, 1136.791785]
    y_coffee = [374.612894, 375.755869, 375.864785, 375.52435, 374.524737]

    peo = len(x_people)
    dis = 0
    for i in range(peo):
        dis += math.sqrt((x_people[i]-x_coffee[0])**2+(y_people[i]-y_coffee[0])**2)
    print(dis, dis/peo)

    dis = 0
    for i in range(peo):
        dis += math.sqrt((x_people[i]-x_coffee[1])**2+(y_people[i]-y_coffee[1])**2)
    print(dis, dis / peo)

    dis = 0
    for i in range(peo):
        dis += math.sqrt((x_people[i] - x_coffee[2]) ** 2 + (y_people[i] - y_coffee[2]) ** 2)
    print(dis, dis / peo)

    dis = 0
    for i in range(peo):
        dis += math.sqrt((x_people[i] - x_coffee[3]) ** 2 + (y_people[i] - y_coffee[3]) ** 2)
    print(dis, dis / peo)

    dis = 0
    for i in range(peo):
        dis += math.sqrt((x_people[i] - x_coffee[4]) ** 2 + (y_people[i] - y_coffee[4]) ** 2)
    print(dis, dis / peo)

def calculate_2():
    d = [[2.965011, 0.964453, 1.009107, 1.343341, 3.100309],
     [0.45845, 1.946267, 2.12008, 1.528449, 0.593749],
     [0.410597, 2.423428, 2.597242, 2.005611, 0.336623],
     [4.731048, 4.389973, 4.563787, 4.504129, 4.866347]]

    for i in range(5):
        dis = 0
        for j in range(4):
            dis += d[j][i]
        print(dis, dis / 4.0)

if __name__ == '__main__':
    calculate_2()