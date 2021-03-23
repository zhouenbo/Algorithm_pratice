#! /usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Enbo Zhou"

import numpy as np

if __name__ == "__main__":
    a = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]
    a = np.array(a)
    print(a)
    # print(a)
    # u, s, vh = np.linalg.svd(a)
    # print(u.shape, s.shape, vh.shape)
    #
    # print(np.diag(s))
    #
    # s = np.hstack((np.diag(s),np.zeros((3,2))))
    # print(s)
    # print(np.dot(np.dot(u,s),vh))
    #

    b = np.ones((3,5))
    print(b)

    # c = np.vstack((b,a))
    # print(c)
    #
    # d = a*b
    # print(d)

    e = np.dot(a,b.T)
    print(e)

    f = np.mean(a,0)
    print(f)

    g = np.array(a,dtype=np.float)
    for i in range(len(g)):
        g[i] -= f
    print(g)
    print(a)

    print(a.ndim)