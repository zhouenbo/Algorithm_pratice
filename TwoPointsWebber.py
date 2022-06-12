#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import copy
import matplotlib.pyplot as plt

__author__ = "Enbo Zhou"


def euclidean_dis(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)


def webber_problem(pts,centers):
    diff = dis_sum = float('inf')
    center_sequence = []
    while diff > 10**(-6):
        center_sequence.append(copy.deepcopy(centers))
        # assignment each point to each center
        dis = 0
        assignments = []
        for pt in pts:
            d1 = euclidean_dis(pt, centers[0])
            d2 = euclidean_dis(pt, centers[1])
            if d1 < d2:
                dis += d1
                assignments.append(0)
            else:
                dis += d2
                assignments.append(1)
        diff = abs(dis_sum - dis)
        dis_sum = dis

        #get new centers
        c1 = [0, 0]
        c2 = [0, 0]
        for i in range(len(assignments)):
            if assignments[i]==0:
                c1[0] += pts[i][0]
                c1[1] += pts[i][1]
            else:
                c2[0] += pts[i][0]
                c2[1] += pts[i][1]
        count = len(assignments) - sum(assignments)
        centers[0][0] = c1[0]/count
        centers[0][1] = c1[1]/count
        centers[1][0] = c2[0]/(len(assignments) - count)
        centers[1][1] = c2[1]/(len(assignments) - count)
    return center_sequence


def plot_res(pts, res):
    x = [pt[0] for pt in pts]
    y = [pt[1] for pt in pts]
    plt.plot(x, y, 'ok', markersize=12, label='Source Points')

    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for centers in res:
        x1.append(centers[0][0])
        y1.append(centers[0][1])
        x2.append(centers[1][0])
        y2.append(centers[1][1])

    #plot the first center trajectory
    plt.plot(x1, y1, 'og', markersize=8, label='Trajectory of Center 1')
    plt.plot(x2, y2, 'ob', markersize=8, label='Trajectory of Center 2')
    plt.plot(x1[-1],y1[-1],'*m', markersize=6, label = 'Center 1')
    plt.plot(x2[-1], y2[-1], '*r', markersize=6, label = 'Center 2')
    plt.legend(loc='best', fontsize=6)
    plt.show()



if __name__ == "__main__":
    pts = [[32,31],[29,32],[27,36],[29,29],[32,29]]
    # centers = [[32,31],[27,36]]
    # res = webber_problem(pts,centers)
    # print(res)

    centers = [[32, 31], [29, 32]]
    res = webber_problem(pts,centers)
    print(res)

    plot_res(pts,res)



