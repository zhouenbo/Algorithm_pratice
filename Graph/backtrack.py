# 489 Robot Room Cleaner
# Description
# Given a robot cleaner in a room modeled as a grid.
#
# Each cell in the grid can be empty or blocked.
#
# The robot cleaner with 4 given APIs can move forward, turn left or turn right. Each turn it made is 90 degrees.
#
# When it tries to move into a blocked cell, its bumper sensor detects the obstacle and it stays on the current cell.
#
# Design an algorithm to clean the entire room using only the 4 given APIs shown below.
#

# """
# This is the robot's control interface.
# You should not implement it, or speculate about its implementation
# """
# class Robot:
#    def move(self):
#        """
#        Returns true if the cell in front is open and robot moves into the cell.
#        Returns false if the cell in front is blocked and robot stays in the current cell.
#        :rtype bool
#        """
#
#    def turnLeft(self):
#        """
#        Robot will stay in the same cell after calling turnLeft/turnRight.
#        Each turn will be 90 degrees.
#        :rtype void
#        """
#
#    def turnRight(self):
#        """
#        Robot will stay in the same cell after calling turnLeft/turnRight.
#        Each turn will be 90 degrees.
#        :rtype void
#        """
#
#    def clean(self):
#        """
#        Clean the current cell.
#        :rtype void
#        """
class Solution:
    """
    :type robot: Robot
    :rtype: None
    """

    def cleanRoom(self, robot):
        # write your code here
        self.cleaned = set()
        direction = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        def dfs(x, y, d):
            robot.clean()
            self.cleaned.add((x, y))
            for i in range(4):
                robot.turnRight()
                d = (d + 1) % 4
                if robot.move():
                    if (x + direction[d][0], y + direction[d][1]) not in self.cleaned:
                        dfs(x + direction[d][0], y + direction[d][1], d)
                    robot.turnLeft()
                    robot.turnLeft()
                    robot.move()
                    robot.turnRight()
                    robot.turnRight()

        dfs(0, 0, 0)


# import requests
# import mysql.connector
# import pandas as pd
#

# I can
# Given an m x n board and a word, find if the word exists in the grid.

# The word can be constructed from letters of sequentially adjacent cells, where "adjacent" cells are horizontally or vertically neighboring. The same #letter cell may not be used more than once.

# Input: board = [["A","B","C","E"],
#                 ["S","F","C","S"],
#                 ["A","D","E","E"]], word = "ABCCED"
# Output: true

board = [["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]]
word = "FCEDF"


def backtrack(board, i, j, visited, k, word):
    row = len(board)
    col = len(board[-1])
    if k == len(word):
        return True

    if board[i][j] != word[k]:
        return False

    if k == len(word) - 1:
        return True

    visited.add((i, j))
    if i > 0 and (i - 1, j) not in visited:
        if backtrack(board, i - 1, j, visited, k + 1, word):
            return True

    if i < row - 1 and (i + 1, j) not in visited:
        if backtrack(board, i + 1, j, visited, k + 1, word):
            return True

    if j > 0 and (i, j - 1) not in visited:
        if backtrack(board, i, j - 1, visited, k + 1, word):
            return True

    if j < col - 1 and (i, j + 1) not in visited:
        if backtrack(board, i, j + 1, visited, k + 1, word):
            return True

    visited.remove((i, j))
    return False


def isExist(board, word):
    row = len(board)
    col = len(board[-1])
    for i in range(row):
        for j in range(col):
            if board[i][j] == word[0] and backtrack(board, i, j, set(), 0, word):
                return True
    return False


print(isExist(board, word))

# MNMN = (MN)^2

