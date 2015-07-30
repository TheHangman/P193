__author__ = 'Michael'
import math
import random
import inverse_shapley
import itertools
# import matplotlib.pyplot as pyplot

def exact_shapley(weights, n_players):
    f = [0]*n_players
    f_plus = [0]*n_players
    sample_size = math.factorial(n_players)
    list_of_perms = list(itertools.permutations(range(1, n_players+1)))
    for m in range(0, sample_size):
        player = list_of_perms[m]
        total = -sum(weights)
        for index in range(0, n_players):
            f[player[index]-1] += math.copysign(1.0, total)
            total += 2*weights[player[index]]
            f_plus[player[index]-1] += math.copysign(1.0, total)
    shapley_value = [((f_plus[x]-f[x])/sample_size) for x in range(0, n_players)]
    return shapley_value

# TEST 1: Generate q_distribution and cumulative
def t1():
    n = random.randint(3, 50)
    distribution = inverse_shapley.q_distribution(n)
    sum_distribution = sum(distribution)
    distribution = [distribution[i]/sum_distribution for i in range(0, len(distribution))]
    cumulative = inverse_shapley.q_cumulative(distribution, n)
    print(sum(distribution))
    # plotting from matplotlib
    # pyplot.plot(cumulative)
    # pyplot.show()
    z = 1000
    K = [0]*z
    for i in range(0, z):
        K[i] = inverse_shapley.sample_k(cumulative, n)
    #pyplot.plot(K)
    #pyplot.show()

# TEST 2: Sample x
def t2():
    for i in range(0, 1000):
        n = random.randint(3, 60)
        k = random.choice(range(1, n))
        X = inverse_shapley.sample_x(k, n)
        assert(sum(X) == (1 + 2*k - n))

# TEST 3: Shapley Value Estimation
def t3():
    epsilon = 0.1
    delta = 0.1
    for j in range(0, 40):
        n = random.randint(3, 8)
        shapley_sample_size = 2*n*math.log(n/(2*delta))/(epsilon**2)
        shapley_sample_size = math.ceil(shapley_sample_size)
        test_weights = [random.random() for i in range(0, n+1)]
        approximate_values = inverse_shapley.estimate_shapley(test_weights, shapley_sample_size, n)
        exact_values = exact_shapley(test_weights, n)
        if inverse_shapley.l2_distance(approximate_values, exact_values) > epsilon:
            print(n)
            print(test_weights)
            print(approximate_values)
            print(exact_values)
            print(inverse_shapley.l2_distance(approximate_values, exact_values))

# TEST 4: Estimate Correlation
def t4():
    for j in range(0, 100):
        n = random.randint(3, 8)
        epsilon = 0.1
        delta = 0.1
        sample_size = math.ceil(2*(n+1)*math.log((n+1)/(2*delta))/(epsilon**2))
        test_weights = [random.random() for i in range(0, n+1)]
        exact_values = exact_shapley(test_weights, n)
        distribution = inverse_shapley.q_distribution(n)
        sum_distribution = sum(distribution)
        distribution = [distribution[i]/sum_distribution for i in range(0, len(distribution))]
        cumulative = inverse_shapley.q_cumulative(distribution, n)
        fcc = inverse_shapley.estimate_correlation(test_weights, sample_size, n, cumulative)
        fcc_avg = sum(fcc[1:n+1])/n
        rhs = [0]*n
        lhs = [0]*n
        for z in range(0, n):
            rhs[z] = 2.0/n + (sum_distribution/2.0)*(fcc[z+1] - fcc_avg)
            lhs[z] = exact_values[z]
        if inverse_shapley.l2_distance(lhs, rhs) > epsilon:
            print(test_weights)
            print(fcc)
            print(exact_values)
            print(lhs)
            print(rhs)
            print(inverse_shapley.l2_distance(lhs, rhs))

# TEST 5: BoostingTTV
def t5():
    for j in range(0, 100):
        n = random.randint(3, 8)
        epsilon = 0.1
        delta = 0.1
        sample_size = math.ceil(2*(n+1)*math.log((n+1)/(2*delta))/((epsilon/16)**2))
        test_weights = [random.random() for i in range(0, n+1)]
        exact_values = exact_shapley(test_weights, n)
        distribution = inverse_shapley.q_distribution(n)
        sum_distribution = sum(distribution)
        distribution = [distribution[i]/sum_distribution for i in range(0, len(distribution))]
        cumulative = inverse_shapley.q_cumulative(distribution, n)
        fcc = inverse_shapley.estimate_correlation(test_weights, sample_size, n, cumulative)
        output_weights = inverse_shapley.boosting_ttv(fcc, epsilon, n, cumulative, sample_size)
        print(inverse_shapley.l2_distance(output_weights,exact_values))
