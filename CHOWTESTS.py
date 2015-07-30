__author__ = 'kae'
import chowreconstruct
import random
import math
import warnings
def generate_sample_weights(n, min_val, max_val):
    weights = [0]*(n+1)
    for x in range(0, n+1):
        weights[x] = random.uniform(min_val, max_val)
    return weights

def exact_chow(g):
    n = len(g)
    beta = [0]*n
    b = [0]*n
    for i in range(0, 2**(n-1)):
        bit_field = int_to_vote_list(i)
        bit_length = len(bit_field)
        a = [1] + [-1]*(n-bit_length-1)+bit_field
        for k in range(0, n):
            b[k] = a[k]*g[k]
        beta[0] += math.copysign(1, sum(b))
        for l in range(1, n):
            beta[l] += math.copysign(1, sum(b))*a[l]
    beta[:] = [x / 2**n for x in beta]
    return beta

def int_to_vote_list(integer):
    bits = [int(bit) for bit in bin(integer)[2:]]
    bits = [1 if x % 2 else -1 for x in bits]
    return bits

# Test 1: truncation operator
def t1():
    input_1 = [0.1, 1, 2, 3, 4, 0.2, -1, -3, 0, 0.5]
    n_1 = 10
    output_1 = chowreconstruct.trunc(input_1, n_1)
    print(output_1)
    assert(output_1 == [0.1, 1, 1, 1, 1, 0.2, -1, -1, 0, 0.5])

# Test 2: l2 distance
def t2():
    input_2 = [1, 2, 3, 4, -1, -3, -0.5, -6]
    input_3 = [-1, -1, -0.5, 6, 2, -3, -0.5, -6]
    output_3 = chowreconstruct.l2_distance(input_2, input_3)
    print(output_3)
    assert(output_3 - 6.18465843843 < 0.1)

# Test 3: integerization
def t3():
    input_4 = [0.5, 0.2, 0.3, 0.6, 0.7, 0.23, 0.62, 0.33, 0.32, 0.11, 0.54]
    epsilon_4 = 0.01
    n_4 = 11
    alpha_4 = [0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    output_4 = chowreconstruct.integerize(input_4, n_4, epsilon_4, alpha_4)
    print(output_4)
    assert(output_4 == [0.5005050441442128, 0.199498743710662, 0.30050504414421286, 0.6005088319990877, 0.7000075757097496, 0.22964987816843838, 0.6201070693966423, 0.3306561786019892, 0.3201032815417675, 0.11055289706022173, 0.5402065630835349])

# Test 4: height
def t4():
    input_5 = [0.5, 0.25, 0.3, 0.45]
    chi_gt_5 = [0.5, 0.125, 0.35, 0.55]
    n_5 = 4
    output_5 = chowreconstruct.half_height(n_5, input_5, chi_gt_5)
    print(output_5)
    assert(output_5 == [0.0/2, 0.125/2, -0.04999999999999999/2, -0.10000000000000003/2])

# Test 5: approximation
def t5():
    input_6 = [0.5, 0.25, 0.3, 0.45]
    m_6 = 14023
    n_6 = 4
    output_6 = chowreconstruct.approximate(input_6, m_6, n_6)
    print(output_6)
    assert(output_6[0] - 0.75 < 0.1)
    assert(output_6[1] - 0.25 < 0.1)
    assert(output_6[1] - 0.25 < 0.1)
    assert(output_6[1] - 0.25 < 0.1)

# Test 6: main algorithm test
def t6():
    input_7 = [0.75, 0.25, 0.25, 0.25]
    epsilon_7 = 0.1
    delta_7 = 0.1
    m_7 = 14023
    n_7 = 4
    output_7 = chowreconstruct.cr1(input_7, epsilon_7, delta_7)
    print(output_7)
    output_8 = chowreconstruct.approximate(output_7, m_7, n_7)
    assert(output_8[0] - input_7[0] < 0.1)
    assert(output_8[1] - input_7[1] < 0.1)
    assert(output_8[2] - input_7[2] < 0.1)
    assert(output_8[3] - input_7[3] < 0.1)

# Test 7: selected chow params
def t7():
    input_8 = [0.75, 0.2, 0.2, 0.1, 0.5]
    epsilon_8 = 0.1
    delta_8 = 0.1
    output_8 = chowreconstruct.cr1(input_8, epsilon_8, delta_8)
    print(output_8)
    print(exact_chow(output_8))

# Test 8: random chow params
def t8():
    weights = generate_sample_weights(4, 0, 10)
    print(weights)
    chow_params = exact_chow(weights)
    print(chow_params)
    output_9 = chowreconstruct.cr1(chow_params, 0.15, 0.1)
    print(output_9)
    c_p_output = exact_chow(output_9)
    print(c_p_output)

# Test all
def t_all():
    t1()
    t2()
    t3()
    t4()
    t5()
    t6()
    t7()
    t8()

def t_long(a, b, x):
    for i in range(a, b+1):
        t_n(i, x)
        print(i)

# Test for n voters and x times
def t_n(n, x):
    for i in range(0, x):
        weights = generate_sample_weights(n, 0, 10)
        chow_params = exact_chow(weights)
        output = chowreconstruct.cr1(chow_params, 0.15, 0.1)
        chow_output = exact_chow(output)
        distance = chowreconstruct.l2_distance(chow_params, chow_output)
        if distance > 0.15:
            warnings.warn("failure")
            print(chow_params)
            print(chow_output)

t_long(4, 8, 100)
