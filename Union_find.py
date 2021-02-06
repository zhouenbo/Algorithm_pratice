#! /usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Enbo Zhou"
from collections import defaultdict


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, x, y):
        self.p[self.find(x)] = self.find(y)


def largestIsland(mat):
    row = len(mat)
    col = len(mat[0])
    dsu = DSU(row*col)
    for i in range(row):
        for j in range(col):
            if i > 0 and mat[i-1][j] == mat[i][j]:
                dsu.union((i-1)*col+j, i*col+j)
            if j > 0 and mat[i][j-1] == mat[i][j]:
                dsu.union(i*col+j-1,i*col+j)
    hashmap = {}
    for i in range(row):
        for j in range(col):
            if mat[i][j] == "1":
                hashmap[dsu.find(i*col+j)] = hashmap.get(dsu.find(i*col+j), 0) + 1
    return max(hashmap.values())

def largestIsland_traversal(mat):
    res = 0
    row = len(mat)
    col = len(mat[0])

    def dfs(i, j):
        area = 1
        mat[i][j] = "0"
        if i > 0 and mat[i-1][j] =="1":
            area += dfs(i-1,j)
        if j>0 and mat[i][j-1]=="1":
            area += dfs(i,j-1)
        if i<row-1 and mat[i+1][j]=="1":
            area += dfs(i+1,j)
        if j<col-1 and mat[i][j+1]=="1":
            area += dfs(i,j+1)
        return area

    for i in range(row):
        for j in range(col):
            if mat[i][j] == "1":
                res = max(res, dfs(i,j))
    return res


if __name__ == "__main__":
    grid = [
        ["1", "1", "1", "1", "0"],
        ["1", "1", "0", "1", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "0", "0", "0"]
    ]

    grid2 = [
        ["1", "1", "0", "0", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "1", "0", "0"],
        ["0", "0", "0", "1", "1"]
    ]

    print(largestIsland(grid))