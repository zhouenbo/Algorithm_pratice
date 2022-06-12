#! /usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Enbo Zhou"


def my_func(file):
    with open(file, 'r') as f:
        for line in f.readlines():
            print(line)

def test_dict():
    a = dict()
    a[3]=3
    a[3]=5

if __name__ == '__main__':
    my_func("myfile.txt")