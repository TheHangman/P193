"""CHOWRECONSTRUCT.py: CHOWRECONSTRUCT finds nearly optimal solutions to the chow parameters problem."""
__author__ = 'Michael'
import math
import random
from operator import add

def cr1(chow_parameters, epsilon, delta):
    # n is the number of chow parameters, index 0 is the chow-0.
    n = len(chow_parameters)
    m = 8*n*math.log(2*n/delta)/(epsilon*epsilon)
    m = int(math.ceil(m))
    g_prime = [0]*n
    g = trunc(g_prime, n)
    chi = approximate(g, m, n)
    chi_g_t = integerize(chi, n, epsilon, chow_parameters)
    rho = l2_distance(chow_parameters, chi_g_t)
    while rho > 4*epsilon:
        g_prime = map(add, g_prime, half_height(n, chow_parameters, chi_g_t))
        g = trunc(g_prime, n)
        chi = approximate(g, m, n)
        chi_g_t = integerize(chi, n, epsilon, chow_parameters)
        rho = l2_distance(chow_parameters, chi_g_t)
    return g

def trunc(g_prime, n):
    for i in range(0, n):
        if g_prime[i] > 1:
            g_prime[i] = 1
        elif g_prime[i] < -1:
            g_prime[i] = -1
    return g_prime

def approximate(g, m, n):
    beta = [0]*n
    a = [0]*n
    b = [0]*n
    for i in range(0, m):
        a[0] = -1
        for j in range(1, n):
            a[j] = random.choice([-1, 1])
        b[0] = g[0]
        for k in range(1, n):
            b[k] = a[k]*g[k]
        sum_b = sum(b)
        beta[0] += math.copysign(1, sum_b)
        for l in range(1, n):
            beta[l] += math.copysign(1, sum_b)*a[l]
    beta[:] = [x / m for x in beta]
    return beta

def integerize(chi, n, epsilon, alpha):
    g_integerized = [0]*n
    epsilon_prime = epsilon/(2*math.sqrt(n))
    for i in range(0, n):
        a = alpha[i] - chi[i]
        b = a % epsilon_prime
        b = round(b/epsilon_prime)
        c = (a // epsilon_prime) + b
        g_integerized[i] = alpha[i] - c*epsilon_prime
    return g_integerized

def l2_distance(x, y):
    length = len(x)
    total = 0
    for i in range(0, length):
        total += ((x[i] - y[i])**2)
    return math.sqrt(total)

def half_height(n, chow_parameters, chi_g_t):
    ht = [0]*n
    for i in range(0, n):
        ht[i] = (chow_parameters[i] - chi_g_t[i])/2
    return ht

