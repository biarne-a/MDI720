#!/bin/usr/python

import math
import sys
import string
import random
import numpy as np
import numpy.linalg as alg


# from sklearn import linear_model


def nextpower(n):
    if n <= 0:
        return 0
    result = 1
    while result < n:
        result = 2 * result
    return result


def pi():
    return "%.9f" % math.pi


def occurences(string):
    dic = {}
    for c in string:
        if c in dic:
            dic[c] += 1
        else:
            dic[c] = 1
    return dic


def shuffle(code, string):
    return ''.join(([code[c] if c in code else c for c in string]))


def dumb_compute_formula(k):
    res = 1
    # for


def quicksort(ll):
    """ a sorting algorithm with a pivot value"""
    if len(ll) <= 1:
        return ll
    else:
        p = ll.pop()
        less = []
        greater = []
        for x in ll:
            if x <= p:
                less.append(x)
            else:
                greater.append(x)
    return quicksort(less) + [p] + quicksort(greater)


def compute_matrix():
    M = np.random.uniform(-1, 1, (5, 6))
    M[:, 0] -= 2 * M[:, 1]
    M[:, 2] -= 2 * M[:, 3]
    M[:, 4] -= 2 * M[:, 5]
    return np.ma.where(M > 0, M, 0)


def compute_matrix2():
    M = np.random.uniform(-1, 1, (20, 5))
    G = M.T.dot(M)
    if np.allclose(G.T, G):
        print("G is symetric")
    else:
        print("G is not symetric")
    eigvals = alg.eigvals(G)
    if np.logical_not(eigvals <= 0).all():
        print("Eigen values are all positive")
    else:
        print("Eigen values are not all positive")
    print("le rang de G est " + str(alg.matrix_rank(G)))
    return eigvals

# M = compute_matrix2()
# print(M)
# print(np.logical_not(M <= 0))

# print(M.logical_not().all())

compute_matrix2()
