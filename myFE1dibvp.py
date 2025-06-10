#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from fn import *


# In[2]:


def myFE1dibvp(a, c, f, p0, QL, u0, L, T, dt, noOfEle, shapeFn):
    xh = np.linspace(0, L, noOfEle + 1)
    h = xh[1] - xh[0]
    noOfIntegPt = 2
    if shapeFn == 1:
        nodes = xh[1:]
    if shapeFn == 2:
        nodes = []
        for e in range(noOfEle):
            x_left = xh[e]
            x_right = xh[e + 1]
            nodes.append((x_left + x_right) / 2) 
            nodes.append(x_right) 
        nodes = np.array(nodes)
    t_steps = np.arange(0, T + dt, dt)
    num_steps = len(t_steps)
    num_nodes = len(nodes)
    W = np.zeros((num_nodes, num_steps))
    uh = []
    W[:, 0] = np.array([u0(x) for x in nodes])
    uh.append(approxSol(W[:, 0], p0, xh, shapeFn))
    M = massM(xh, shapeFn, noOfIntegPt)
    K = stiffK(a, c, xh, shapeFn, noOfIntegPt)
    A = M + 0.5 * dt * K
    B = M - 0.5 * dt * K
    for step in range(1, num_steps):
        t_prev = t_steps[step - 1]
        t_current = t_steps[step]
        QL_prev = QL(t_prev)
        F_prev = loadF(a, c, lambda x: f(x, t_prev), p0, QL_prev, xh, shapeFn, noOfIntegPt)
        QL_current = QL(t_current)
        F_current = loadF(a, c, lambda x: f(x, t_current), p0, QL_current, xh, shapeFn, noOfIntegPt)
        rhs = B.dot(W[:, step - 1]) + 0.5 * dt * (F_prev + F_current)
        W[:, step] = spsolve(A.tocsr(), rhs)
        uh.append(approxSol(W[:, step], p0, xh, shapeFn))    
    return uh, W

