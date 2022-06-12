#! /usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Enbo Zhou"


class Node:
    def __init__(self, val = 0):
        self.val = val
        self.children = []


def topological_sort(node, visited):
    if not node:
        return
    visited.add(node)
    for child in node.children:
        if child not in visited:
            topological_sort(child, visited)
    res.append(node.val)


if __name__ == "__main__":
    g = []
    for i in range(10):
        g.append(Node(i))
        g[i].children.extend(g[0:i])

    res = []
    visited = set()
    for n in g:
        if n not in visited:
            topological_sort(n, visited)

    print(res[::-1])

