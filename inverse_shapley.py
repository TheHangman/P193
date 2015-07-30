__author__ = 'Michael'
import random
import math

def inverse_shapley_main(target_shapley, epsilon, delta):
    n = len(target_shapley)
    fcc = [0]*(n+1)
    distribution = q_distribution(n)
    sum_distribution = sum(distribution)
    distribution = [distribution[i]/sum_distribution for i in range(0, n-1)]
    cumulative = q_cumulative(distribution, n)
    g = 1/((n+1)*(2.0**(1/epsilon)))
    fcc_sample_size = 512*(n+1)*math.log((n+1)/(2*delta))/(epsilon**2)
    shapley_sample_size = 800*n*math.log(n/(2*delta))/(g**2)
    shapley_sample_size = math.ceil(shapley_sample_size)
    fcc_sample_size = math.ceil(fcc_sample_size)
    output_games = set()
    for fcc_0_point in float_range(-1.0, 1.0, g):
        for fcc_avg_point in float_range(-1.0, 1.0, g):
            fcc[0] = fcc_0_point
            for i in range(1, n+1):
                fcc[i] = fcc_avg_point + (2.0/sum_distribution)*(target_shapley[i] - (2.0/n))
            h = boosting_ttv(fcc, g, n, cumulative, fcc_sample_size)
            h_shapley = estimate_shapley(h, shapley_sample_size, n)
            if l2_distance(h_shapley, target_shapley) < 8*epsilon/10:
                output_games.add(h)
    return output_games

def boosting_ttv(a_l, eta, n, cumulative, sample_size):
    gamma = eta/2
    trial_weights = [0]*(n+1)
    t = 0
    idx_array = set()
    while (t == 0)or(idx_array != set()):
        a_lt = estimate_correlation(trial_weights, sample_size, n, cumulative)
        idx_array = failed_idx(a_l, a_lt, n, gamma)
        if idx_array == set():
            l = random.sample(idx_array, 1)
            if a_l[l] - a_lt[l] > gamma:
                trial_weights[l] += 1.0
            else:
                trial_weights[0] -= gamma
            trial_weights = [trial_weights[i]*gamma for i in range(0, n+1)]
        trial_weights = trunc(trial_weights, n)
        t += 1
    return trial_weights

def estimate_shapley(weights, sample_size, n):
    f = [0]*n
    f_plus = [0]*n
    for m in range(0, sample_size):
        player = random.sample(range(1, n+1), n)
        total = -sum(weights)
        for i in range(0, n):
            f[player[i]-1] += math.copysign(1.0, total)
            total += 2*weights[player[i]]
            f_plus[player[i]-1] += math.copysign(1.0, total)
    shapley_value = [((f_plus[x]-f[x])/sample_size) for x in range(0, n)]
    return shapley_value

def estimate_correlation(weights, sample_size, n, cumulative):
    correlation = [0]*(n+1)
    for m in range(0, sample_size):
        k = sample_k(cumulative, n)
        x = sample_x(k, n)
        f = [x[i]*weights[i] for i in range(0, n+1)]
        total = sum(f)
        correlation[0] += math.copysign(1.0, total)
        for l in range(1, n+1):
            correlation[l] += math.copysign(1.0, total)*x[l]
    correlation = [correlation[i]/sample_size for i in range(0, n+1)]
    return correlation

def failed_idx(a, b, n, gamma):
    idx_set = set()
    for i in range(0, n+1):
        if a[i] - b[i] > gamma:
            idx_set.add(i)
    return idx_set

def q_distribution(n):
    distribution = [0]*(n-1)
    for k in range(0, n-1):
        distribution[k] = 1/(k+1) + 1/(n-k-1)
    return distribution

def q_cumulative(distribution, n):
    cumulative = [distribution[0]]
    for k in range(1, n-1):
        cumulative.append(distribution[k])
        cumulative[k] += cumulative[k-1]
    return cumulative

def sample_k(cumulative, n):
    r = random.random()
    k = 0
    while r > cumulative[k]:
        k += 1
    # If statement to overcome very unlikely float comparisons ie. 0.999999 > 0.999998
    if k == n-1:
        k -= 1
    return k+1

def sample_x(k, n):
    x = [-1]*(n+1)
    omega = range(1, n+1)
    indices = random.sample(omega, k)
    for i in indices:
        x[i] = 1
    return x

def trunc(weights, n):
    for i in range(0, n):
        if weights[i] > 1:
            weights[i] = 1
        elif weights[i] < -1:
            weights[i] = -1
    return weights

def l2_distance(x, y):
    length = len(x)
    total = 0
    for i in range(0, length):
        total += ((x[i] - y[i])**2)
    return math.sqrt(total)

def float_range(start, stop, step):
    while start < stop:
        yield start
        start += step
