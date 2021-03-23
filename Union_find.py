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






# 434. Number of Islands II
# 中文English
# Given a n,m which means the row and column of the 2D matrix and an array of pair A( size k). Originally, the 2D matrix is all 0 which means there is only sea in the matrix. The list pair has k operator and each operator has two integer A[i].x, A[i].y means that you can change the grid matrix[A[i].x][A[i].y] from sea to island. Return how many island are there in the matrix after each operator.
#
# Example
# Example 1:
#
# Input: n = 4, m = 5, A = [[1,1],[0,1],[3,3],[3,4]]
# Output: [1,1,2,2]
# Explanation:
# 0.  00000
#     00000
#     00000
#     00000
# 1.  00000
#     01000
#     00000
#     00000
# 2.  01000
#     01000
#     00000
#     00000
# 3.  01000
#     01000
#     00000
#     00010
# 4.  01000
#     01000
#     00000
#     00011
# Example 2:
#
# Input: n = 3, m = 3, A = [[0,0],[0,1],[2,2],[2,1]]
# Output: [1,1,2,2]
# Notice
# 0 is represented as the sea, 1 is represented as the island. If two 1 is adjacent, we consider them in the same island. We only consider up/down/left/right adjacent.


"""
Definition for a point.
class Point:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b
"""


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        if x != self.p[x]:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, x, y):
        self.p[self.find(x)] = self.find(y)


class Solution:
    """
    @param n: An integer
    @param m: An integer
    @param operators: an array of point
    @return: an integer array
    """

    def numIslands2(self, n, m, operators):
        # write your code here
        if not operators:
            return []

        dsu = DSU(len(operators))
        visited = {(operators[0].x, operators[0].y): 0}
        res = [1]

        for i in range(1, len(operators)):
            if (operators[i].x, operators[i].y) not in visited:
                count = set()
                if (operators[i].x - 1, operators[i].y) in visited:
                    count.add(dsu.find(visited[(operators[i].x - 1, operators[i].y)]))
                    dsu.union(i, visited[(operators[i].x - 1, operators[i].y)])
                if (operators[i].x + 1, operators[i].y) in visited:
                    count.add(dsu.find(visited[(operators[i].x + 1, operators[i].y)]))
                    dsu.union(i, visited[(operators[i].x + 1, operators[i].y)])
                if (operators[i].x, operators[i].y - 1) in visited:
                    count.add(dsu.find(visited[(operators[i].x, operators[i].y - 1)]))
                    dsu.union(i, visited[(operators[i].x, operators[i].y - 1)])
                if (operators[i].x, operators[i].y + 1) in visited:
                    count.add(dsu.find(visited[(operators[i].x, operators[i].y + 1)]))
                    dsu.union(i, visited[(operators[i].x, operators[i].y + 1)])
                visited[(operators[i].x, operators[i].y)] = i
                res.append(res[-1] - len(count) + 1)
            else:
                res.append(res[-1])
        return res