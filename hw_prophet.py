import numpy as np 
from scipy.stats import truncnorm, uniform, expon, norm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
sns.set(style='darkgrid')

np.random.seed(0)

prob = 'truncnorm'
l, h = 1, 100
n_arr = [10, 20, 40, 60, 80, 100, 500]#100, 500, 1000, 5000, 10000]
k = 2
eta = 0

k_arr = [1, 10, 25, 50]
N = 100

def plot_perf(n_arr, performance, ratio, dist):
    plt.xlabel('Number of samples (n)')
    plt.ylabel(r"$E[ALG]/E[X_{max}]$")
    plt.title('ALG performance against the prophet vs n')
    plt.axhline(y=ratio, color='g', linestyle='--')
    plt.plot(n_arr, performance, marker='o', label=dist)
    plt.legend(bbox_to_anchor=(0.9,0.2),loc=4, borderaxespad=0., frameon=True, framealpha=0.6, fontsize='small')
    plt.show()

def plotk(k_arr, performance, dist):
    plt.xlabel('Items selected (k)')
    plt.ylabel(r"$E[ALG]/E[X_{max}]$")
    plt.title('ALG performance against the prophet vs k')
    plt.plot(k_arr, performance, marker='o', label='n = '+str(N), color='purple')
    plt.axhline(y=0.25, color='g', linestyle='--')
    # plt.legend(bbox_to_anchor=(0.9,0.2),loc=4, borderaxespad=0., frameon=True, framealpha=0.6, fontsize='small')
    plt.show()

def get_gauss_ab(h):
    my_mean = np.random.uniform() * h
    my_std = np.random.uniform()
    samples = sorted(norm.rvs(my_mean, my_std, size=2))
    a, b = samples[0], samples[1]

    return a, b, my_mean, my_std

def get_sample(l, h, dist):
    a = np.random.randint(low=l, high=h)
    b = np.random.randint(low=a, high=h)
    if dist == 'uniform':
        x = np.random.uniform(a, b)
        mean_x = (b+a) * 0.5
        median_x = mean_x
        params = [a, b]
    elif dist == 'truncnorm':
        a, b, my_mean, my_std = get_gauss_ab(h)
        x = truncnorm.rvs(a, b, loc=my_mean, scale=my_std)
        mean_x = truncnorm.mean(a, b)
        median_x = truncnorm.median(a, b)
        params = [a, b, my_mean, my_std]
    elif dist == 'expon':
        x = expon.rvs(loc=a, scale=1./b)
        mean_x = expon.mean(loc=a, scale=1./b)
        median_x = expon.median(loc=a, scale=1./b)
        params = [a, 1./b]
        
    return x, mean_x, median_x, params

def select(x, thresh, k):
    if k == 1:
        ret = 1 if x > thresh else 0
    else:
        rn = np.random.uniform()
        if rn < 0.5:
            ret = 0
        else:
            ret = 1 if x > thresh else 0

    return ret

def get_top_k(x_arr, mean_x_arr, k):
    d = dict()
    for x, mean_x in zip(x_arr, mean_x_arr):
        d[x] = mean_x
    d = dict(reversed(sorted(d.items())))

    mean_max_x = []
    c = 0
    for v in d.values():
        mean_max_x.append(v)
        c += 1
        if c==k:
            return sum(mean_max_x)

def get_thresh(params, dist, k, p=0.25):
    p = 1./(2. * k)
    if dist == 'uniform':
        a, b = params[0], params[1]
        tau = uniform.ppf(p, loc=a, scale=b)
    elif dist == 'truncnorm':
        a, b, my_mean, my_std = params[0], params[1], params[2], params[3]
        tau = truncnorm.ppf(p, a=a, b=b, loc=my_mean, scale=my_std)

    return tau


def n_vs_alg():
    performance = []
    for n in n_arr:
        x_arr = []
        mean_xarr = []
        median_xarr = []
        dist_params = []
        for _ in range(n):
            x, mean_x, median_x, param = get_sample(l, h, dist=prob)
            x_arr.append(x)
            mean_xarr.append(mean_x)
            median_xarr.append(median_x)
            dist_params.append(param)

        if k == 1:
            idx = np.argmax(x_arr)
            mean_max_x = mean_xarr[idx]
            if eta:
                thresh = get_thresh(dist_params[idx], prob, k)
            else:
                thresh = 0.5 * mean_max_x
        elif k == 2:
            mean_max_x = get_top_k(x_arr, mean_xarr, 2)

        mean_alg_x = -1
        alg_x = -1
        n_selected = 0
        alg_perf = 0
        for i in range(len(x_arr)):
            if k > 1:
                thresh = get_thresh(dist_params[i], prob, k)
            selected = select(x_arr[i], thresh, k=1)
            if selected:
                n_selected += 1
                alg_x = x_arr[i]
                alg_perf += mean_xarr[i]
                if n_selected >= k:
                    performance.append(alg_perf/mean_max_x)
                    break
        if n_selected==0:
            alg_perf += mean_xarr[-1]
            performance.append(alg_perf/mean_max_x)
        
        # print("x arr: ", x_arr)
        # print("mean: ", mean_xarr)
        # print("median: ", median_xarr)
        # print("threshold: ", thresh)
        # print(alg_perf, mean_max_x)
        # print(performance)

    plot_perf(n_arr, performance, 0.25, prob)


def k_vs_alg():
    x_arr = []
    mean_xarr = []
    median_xarr = []
    dist_params = []
    for _ in range(N):
        x, mean_x, median_x, param = get_sample(l, h, dist=prob)
        x_arr.append(x)
        mean_xarr.append(mean_x)
        median_xarr.append(median_x)
        dist_params.append(param)
    
    performance = []
    for k in k_arr:
        if k == 1:
            idx = np.argmax(x_arr)
            mean_max_x = mean_xarr[idx]
            if eta:
                thresh = get_thresh(dist_params[idx], prob, k)
            else:
                thresh = 0.5 * mean_max_x
        else:
            mean_max_x = get_top_k(x_arr, mean_xarr, k)

        mean_alg_x = -1
        alg_x = -1
        n_selected = 0
        alg_perf = 0
        for i in range(len(x_arr)):
            if k > 1:
                thresh = get_thresh(dist_params[i], prob, k)
            selected = select(x_arr[i], thresh, k=k)
            if selected:
                n_selected += 1
                alg_x = x_arr[i]
                alg_perf += mean_xarr[i]
                if n_selected >= k:
                    performance.append(alg_perf/mean_max_x)
                    break

        if n_selected<k:
            to_select = k - n_selected
            for j in range(len(mean_xarr), -1, -1):
                alg_perf += mean_xarr[-1]
                to_select -= 1
                if to_select == 0:
                    performance.append(alg_perf/mean_max_x)
        
    plotk(k_arr, performance, prob)



def main():
    # n_vs_alg()
    k_vs_alg()
    


if __name__ == '__main__':
    main()