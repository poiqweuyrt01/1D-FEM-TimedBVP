#!/usr/bin/env python
# coding: utf-8


import numpy as np
from scipy.sparse import lil_matrix



def shapeFn1d(i, x, x1, x2, p):
    x = np.asarray(x)
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y = np.zeros_like(x)
    mask = (x >= x1) & (x <= x2) 
    if p == 1:
        if i == 1:
            y[mask] = (x2[mask] - x[mask]) / (x2[mask] - x1[mask])
        elif i == 2:
            y[mask] = (x[mask] - x1[mask]) / (x2[mask] - x1[mask])
    elif p == 2:
        if i == 1:
            y[mask] = ((x2[mask] - x[mask]) * (x2[mask] - 2 * x[mask] + x1[mask])) / ((x2[mask] - x1[mask])**2)
        elif i == 2:
            y[mask] = 4 * ((x[mask] - x1[mask]) * (x2[mask] - x[mask])) / ((x2[mask] - x1[mask])**2)
        elif i == 3:
            y[mask] = ((x1[mask] - x[mask]) * (x2[mask] - 2 * x[mask] + x1[mask])) / ((x2[mask] - x1[mask])**2)
    return y



def shapeFnDer1d(i, x, x1, x2, p):
    if x < x1 or x > x2:
        return 0.0
    if p == 1:
        if i == 1:
            y = -1 / (x2 - x1)
        elif i == 2:
            y = 1 / (x2 - x1)
    elif p == 2:
        if i == 1:
            y = (4 * x - x1 - 3 * x2) / ((x2 - x1)**2)
        elif i == 2:
            y = (4 * (x1 + x2 - 2 * x)) / ((x2 - x1)**2)
        elif i == 3:
            y = (4 * x - 3 * x1 - x2) / ((x2 - x1)**2)
    return y



def gaussQuadStd1d(g, noOfIntegPt):
    if noOfIntegPt == 2:
        y = g(-1/((3)**0.5)) + g(1/((3)**0.5))
    if noOfIntegPt == 3:
        y = 5/9 * g(-(3/5)**0.5) + 8/9 * g(0) + 5/9 * g((3/5)**0.5)
    return y



def gaussQuad1d(fn, lowerLimit, upperLimit, noOfIntegPt):
    def g(xi):
        x = 0.5 * (upperLimit - lowerLimit) * (xi + 1) + lowerLimit
        return fn(x) * (upperLimit - lowerLimit) / 2.0
    y = gaussQuadStd1d(g, noOfIntegPt)
    return y


def meij(e, i, j, xh, shapeFn, noOfIntegPt):
    x1 = xh[e-1]    
    x2 = xh[e]
    def fn(x):
        psi_i = shapeFn1d(i, x, x1, x2, shapeFn)
        psi_j = shapeFn1d(j, x, x1, x2, shapeFn)
        g = psi_i * psi_j
        return g
    y = gaussQuad1d(fn, x1, x2, noOfIntegPt)
    return y



def keij(a, c, e, i, j, xh, shapeFn, noOfIntegPt):
    x1 = xh[e-1]
    x2 = xh[e]
    def fn(x):
        psi_i = shapeFn1d(i, x, x1, x2, shapeFn)
        psi_j = shapeFn1d(j, x, x1, x2, shapeFn)
        dpsi_i = shapeFnDer1d(i, x, x1, x2, shapeFn)
        dpsi_j = shapeFnDer1d(j, x, x1, x2, shapeFn)
        a_val = a(x)
        c_val = c(x)
        g = a_val * dpsi_i * dpsi_j + c_val * psi_i * psi_j
        return g
    y = gaussQuad1d(fn, x1, x2, noOfIntegPt)
    return y



def fei(a, c, f, p0, e, i, xh, shapeFn, noOfIntegPt):
    x1 = xh[e-1]
    x2 = xh[e]
    def fn(x):
        psi_i = shapeFn1d(i, x, x1, x2, shapeFn)
        psi_i_1 = shapeFn1d(1, x, xh[0], xh[1], shapeFn)
        dpsi_i = shapeFnDer1d(i, x, x1, x2, shapeFn)
        dpsi_i_1 = shapeFnDer1d(1, x, xh[0], xh[1], shapeFn)
        a_val = a(x)
        c_val = c(x)
        f_val = f(x)
        g = f_val * psi_i - a_val * p0 * dpsi_i_1 * dpsi_i - c_val * p0 * psi_i_1 * psi_i
        return g
    y = gaussQuad1d(fn, x1, x2, noOfIntegPt)
    return y



def massM(xh, shapeFn, noOfIntegPt):
    if shapeFn == 1:
        num_nodes = len(xh) - 1
    elif shapeFn == 2:
        num_nodes = 2 * len(xh) - 2
    M = lil_matrix((num_nodes, num_nodes), dtype=float)
    for e in range(1, len(xh)):
        if shapeFn == 1:
            global_nodes = [e - 1, e]
        if shapeFn == 2:
            global_nodes = [2 * (e - 1), 2 * (e - 1) + 1, 2 * (e - 1) + 2]
        if e == 1:
            for i in range(shapeFn): 
                for j in range(shapeFn):
                    m_ij = meij(e, i + 2, j + 2, xh, shapeFn, noOfIntegPt)
                    M[i, j] = M[i, j] + m_ij
        if e > 1: 
            for i in range(shapeFn + 1): 
                for j in range(shapeFn + 1): 
                    m_ij = meij(e, i + 1, j + 1, xh, shapeFn, noOfIntegPt)
                    i_global = global_nodes[i] - 1
                    j_global = global_nodes[j] - 1
                    M[i_global, j_global] = M[i_global, j_global] + m_ij
    return M.tocsr()



def stiffK(a, c, xh, shapeFn, noOfIntegPt):
    if shapeFn == 1:
        num_nodes = len(xh) - 1
    elif shapeFn == 2:
        num_nodes = 2 * len(xh) - 2
    K = lil_matrix((num_nodes, num_nodes), dtype=float)
    for e in range(1, len(xh)):
        if shapeFn == 1:
            global_nodes = [e - 1, e]
        if shapeFn == 2:
            global_nodes = [2 * (e - 1), 2 * (e - 1) + 1, 2 * (e - 1) + 2]
        if e == 1:
            for i in range(shapeFn): 
                for j in range(shapeFn):
                    k_ij = keij(a, c, e, i + 2, j + 2, xh, shapeFn, noOfIntegPt)
                    K[i, j] = K[i, j] + k_ij
        if e > 1: 
            for i in range(shapeFn + 1): 
                for j in range(shapeFn + 1): 
                    k_ij = keij(a, c, e, i + 1, j + 1, xh, shapeFn, noOfIntegPt)
                    i_global = global_nodes[i] - 1
                    j_global = global_nodes[j] - 1
                    K[i_global, j_global] = K[i_global, j_global] + k_ij
    return K.tocsr()



def loadF(a, c, f, p0, QL, xh, shapeFn, noOfIntegPt):
    if shapeFn == 1:
        num_nodes = len(xh) - 1
    elif shapeFn == 2:
        num_nodes = 2 * len(xh) - 2
    F = np.zeros(num_nodes)
    for e in range(1, len(xh)):
        if shapeFn == 1:
            global_nodes = [e - 1, e]
        if shapeFn == 2:
            global_nodes = [2 * (e - 1),2 * (e - 1) + 1, 2 * (e - 1) + 2]
        if e == 1:
            for i in range(shapeFn): 
                f_i = fei(a, c, f, p0, e, i + 2, xh, shapeFn, noOfIntegPt)
                F[i] = F[i] + f_i
        if e > 1:
            for i in range(shapeFn + 1): 
                f_i = fei(a, c, f, p0, e, i + 1, xh, shapeFn, noOfIntegPt)
                i_global = global_nodes[i] - 1
                F[i_global] = F[i_global] + f_i
    F[-1] = F[-1] + QL 
    return F



def L2norm1d(f, a, b, noOfEle):
    h = (b - a) / noOfEle 
    total = 0.0
    xi = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    w = np.array([5/9, 8/9, 5/9])  
    for i in range(noOfEle):
        x_left = a + i * h 
        x_right = x_left + h
        for k in range(3):
            x = (x_right + x_left) / 2 + (xi[k] * h / 2)
            fx_sq = (f(x)) ** 2 
            total = total + w[k] * fx_sq  
    y = total * h / 2 
    return np.sqrt(y)  



def approxSol(w: np.array, p0: float, xh: np.array, shapeFn:int):
    w = np.hstack((p0, w))
    if (shapeFn == 1): 
        uh = lambda x: linearApprox(x,w,xh)
    else:
        uh = lambda x: quadraticApprox(x,w,xh)
    return uh
    


def linearApprox(x: np.array, v: np.array,xh: np.array):
    if (np.ndim(x) == 0):
        x = np.array([x])
    y = np.zeros(len(x))
    h = xh[1]-xh[0]
    ind = np.floor(x/h).astype(int)
    x1 = h * ind
    x2 = x1 + h
    ind = ind+1
    indLastPt = np.where(ind == len(v))[0]
    y[indLastPt] = v[-1]
    indNotLastPt = np.where(ind < len(v))[0]
    y[indNotLastPt] = v[ind[indNotLastPt]-1] * shapeFn1d(1,x[indNotLastPt],x1[indNotLastPt],x2[indNotLastPt],1) + v[ind[indNotLastPt]] * shapeFn1d(2,x[indNotLastPt],x1[indNotLastPt],x2[indNotLastPt],1)
    return y



def quadraticApprox(x: np.array,v: np.array,xh: np.array):
    if (np.ndim(x) == 0):
        x = np.array([x])
    y = np.zeros(len(x))
    h = xh[1]-xh[0]
    ind = np.floor(x/h).astype(int)
    x1 = ind*h
    x2 = x1+h
    ind = ind+1
    indLastPt = np.where(ind == (len(v)+1)/2)[0]
    y[indLastPt] = v[-1]
    indNotLastPt = np.where(ind < (len(v)+1)/2)[0]
    y[indNotLastPt] = v[2*ind[indNotLastPt]-2] * shapeFn1d(1,x[indNotLastPt],x1[indNotLastPt],x2[indNotLastPt],2) + v[2*ind[indNotLastPt]-1] * shapeFn1d(2,x[indNotLastPt],x1[indNotLastPt],x2[indNotLastPt],2) + v[2*ind[indNotLastPt]] * shapeFn1d(3,x[indNotLastPt],x1[indNotLastPt],x2[indNotLastPt],2)
    return y

