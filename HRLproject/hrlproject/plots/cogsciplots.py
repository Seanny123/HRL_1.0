"""Plots for CogSci 2014 paper."""

import os, random, copy

import pylab as plt
import numpy as np

def load_data(prefix, num_files):
    data = []
    for i in range(num_files):
        with open(os.path.join(prefix, "dataoutput_%s.txt" % i)) as f:
            data += [[[[float(v) for v in entry.split(" ")] for entry in r.split(";")] for r in f.readlines()]]
    return data

def get_sum(data):
    timestep = data[-1][0] - data[-2][0]
    sum = [[0, 0]]
    for i, x in enumerate(data):
        sum += [[x[0], sum[i - 1][1] + x[1] * timestep]]
    return sum


def get_avg(data):
    filter = 1e-1
    avg = [[0, 0]]
    for i, x in enumerate(data):
        avg += [[x[0], filter * x[1] + (1 - filter) * avg[i - 1][1]]]
    return avg

def pad(data):
    max_len = max([len(row) for row in data])
    newdata = copy.deepcopy(data)
    for i, row in enumerate(newdata):
        newdata[i] += [["NA" for _ in newdata[i][0]] for _ in range(max_len - len(row))]
    return newdata


def transpose(m):
    m = pad(m)
    return [[row[i] for row in m] for i in range(len(m[0]))]


def sample(data):
    for _ in data:
        yield random.choice(data)


def bootstrapci(data, n, p, func):
    data = [d for d in data if d != "NA"]
#    print data
    index = int(n * (1 - p) / 2)
    r = [list(sample(data)) for _ in range(n)]
    r = [func(x) for x in r]
    r.sort()
    return r[index], r[-index]


def hierarchical_vs_flat():
    """Comparing system with hierarchy vs flat system on delivery task."""

    flat = load_data(os.path.join("..", "..", "data", "delivery", "flat"), 5)
    untrained = load_data(os.path.join("..", "..", "data", "delivery", "untrained"), 5)
    rand = load_data(os.path.join("..", "..", "data", "delivery", "random"), 1)
    optimal = load_data(os.path.join("..", "..", "data", "delivery", "optimal"), 1)

    flat = flat[1:] # flat_0 is old data atm

    timestep = flat[0][0][-1][0] - flat[0][0][-2][0]

    # calculate sums
    flat_sums = []
    for data in flat:
        flat_sums += [get_sum(data[0])]

    untrained_sums = []
    for data in untrained:
        untrained_sums += [get_sum(data[0])]

    rand = get_sum(rand[0][0])
    optimal = get_sum(optimal[0][0])

    # fit random line
    r_b, r_c = np.polyfit([x[0] for x in rand], [x[1] for x in rand], 1)

    # subtract the random baseline
    for s in flat_sums + untrained_sums + [optimal]:
        for i, _ in enumerate(s):
            s[i][1] -= r_b * s[i][0] + r_c

    # calculate means
    flat_mean = []
    for s in transpose(flat_sums):
        step = [v[1] for v in s if v[1] != "NA"]
        flat_mean += [sum(step) / len(step)]
    untrained_mean = []
    for s in transpose(untrained_sums):
        step = [v[1] for v in s if v[1] != "NA"]
        untrained_mean += [sum(step) / len(step)]

    # fit lines
    o_b, o_c = np.polyfit([x[0] for x in optimal],
                          [x[1] for x in optimal], 1)
    f_b, f_c = np.polyfit([t * timestep for t in range(100)], flat_mean[1000:1100], 1)
    u_b, u_c = np.polyfit([t * timestep for t in range(100)], untrained_mean[1000:1100], 1)

    print "o_b", o_b
    print "f_b", f_b, f_b / o_b
    print "u_b", u_b, u_b / o_b

    # confidence intervals
    flat_cis = [bootstrapci([v[1] for v in s], 1000, 0.95,
                                  lambda x: sum(x) / len(x))
                      for s in transpose(flat_sums)]
    untrained_cis = [bootstrapci([v[1] for v in s], 1000, 0.95,
                                  lambda x: sum(x) / len(x))
                      for s in transpose(untrained_sums)]

    fig = plt.figure(figsize=[6, 4])
    ax = plt.axes([0.13, 0.13, 0.75, 0.75])
    for s in flat_sums:
        plt.plot([x[0] for x in s], [x[1] for x in s], "-", color='red', linewidth=1, alpha=0.4)
#    for s in untrained_sums:
#        plt.plot([x[0] for x in s], [x[1] for x in s], "-", color='green', linewidth=1, alpha=0.4)

    plt.plot([t * timestep for t in range(len(flat_mean))], flat_mean, color="red", linewidth=3,
             label="flat")
    plt.plot([t * timestep for t in range(len(untrained_mean))], untrained_mean, color="green", linewidth=3,
             label="hierarchical")

    plt.fill_between([t * timestep for t in range(len(flat_cis))],
                     [x[0] for x in flat_cis],
                     [x[1] for x in flat_cis],
                     color="red", alpha=0.3)
    plt.fill_between([t * timestep for t in range(len(untrained_cis))],
                     [x[0] for x in untrained_cis],
                     [x[1] for x in untrained_cis],
                     color="green", alpha=0.3)
    trange = [t * timestep for t in range(2000)]
    plt.plot(trange, [o_b * t + o_c for t in trange], "-.", color='black', linewidth=3, label="optimal")
    plt.xlabel("time (s)")
    plt.ylabel("accumulated reward")
    plt.legend(borderpad=0.3, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_yaxis().tick_left()
    ax.get_xaxis().tick_bottom()
    plt.xlim([0, 5500])
    plt.ylim([0, 250])

    plt.savefig(r"D:\Documents\officesvn\writing\cogsci2014\figures\hier_vs_flat.png")
    plt.savefig(r"D:\Documents\officesvn\writing\cogsci2014\figures\hier_vs_flat.pdf")

def pretrained_vs_nopretrained():
    """Comparing system with pretraining on context task vs one without."""

    pretrained = load_data(os.path.join("..", "..", "data", "delivery", "pretrained"), 5)
    untrained = load_data(os.path.join("..", "..", "data", "delivery", "untrained"), 5)
    rand = load_data(os.path.join("..", "..", "data", "delivery", "random"), 1)
    optimal = load_data(os.path.join("..", "..", "data", "delivery", "optimal"), 1)

    pretrained = pretrained[1:] # pretrained 0 is old data at the moment

    timestep = pretrained[0][0][-1][0] - pretrained[0][0][-2][0]

    pretrained_sums = []
    for data in pretrained:
        pretrained_sums += [get_sum(data[0])]

    untrained_sums = []
    for data in untrained:
        untrained_sums += [get_sum(data[0])]

    rand = get_sum(rand[0][0])
    optimal = get_sum(optimal[0][0])

    # fit random line
    r_b, r_c = np.polyfit([x[0] for x in rand], [x[1] for x in rand], 1)

    # subtract the random baseline
    for s in pretrained_sums + untrained_sums + [optimal]:
        for i, _ in enumerate(s):
            s[i][1] -= r_b * s[i][0] + r_c

    # means
    pretrained_mean = []
    for s in transpose(pretrained_sums):
        step = [v[1] for v in s if v[1] != "NA"]
        pretrained_mean += [sum(step) / len(step)]
    untrained_mean = []
    for s in transpose(untrained_sums):
        step = [v[1] for v in s if v[1] != "NA"]
        untrained_mean += [sum(step) / len(step)]

    # fit lines line
    o_b, o_c = np.polyfit([x[0] for x in optimal],
                          [x[1] for x in optimal], 1)
    p_b, p_c = np.polyfit([t * timestep for t in range(100)], pretrained_mean[1000:1100], 1)
    u_b, u_c = np.polyfit([t * timestep for t in range(100)], untrained_mean[1000:1100], 1)

    print "o_b", o_b
    print "p_b", p_b, p_b / o_b
    print "u_b", u_b, u_b / o_b

    # confidence intervals
    pretrained_cis = [bootstrapci([v[1] for v in s], 1000, 0.95,
                                  lambda x: sum(x) / len(x))
                      for s in transpose(pretrained_sums)]
    untrained_cis = [bootstrapci([v[1] for v in s], 1000, 0.95,
                                  lambda x: sum(x) / len(x))
                      for s in transpose(untrained_sums)]

    fig = plt.figure(figsize=[6, 4])
    ax = plt.axes([0.13, 0.13, 0.75, 0.75])
#    for s in pretrained_sums:
#        plt.plot([x[0] for x in s], [x[1] for x in s], "-", color='red', linewidth=1, alpha=0.4)
#    for s in untrained_sums:
#        plt.plot([x[0] for x in s], [x[1] for x in s], "-", color='green', linewidth=1, alpha=0.4)

    plt.plot([t * timestep for t in range(len(pretrained_mean))], pretrained_mean, color="red", linewidth=3,
             label="transfer")
    plt.plot([t * timestep for t in range(len(untrained_mean))], untrained_mean, color="green", linewidth=3,
             label="no transfer")

    plt.fill_between([t * timestep for t in range(len(pretrained_cis))],
                     [x[0] for x in pretrained_cis],
                     [x[1] for x in pretrained_cis],
                     color="red", alpha=0.3)
    plt.fill_between([t * timestep for t in range(len(untrained_cis))],
                     [x[0] for x in untrained_cis],
                     [x[1] for x in untrained_cis],
                     color="green", alpha=0.3)
    trange = [t * timestep for t in range(2000)]
    plt.plot(trange, [o_b * t + o_c for t in trange], "-.", color='black', linewidth=3, label="optimal")
    plt.xlabel("time (s)")
    plt.ylabel("accumulated reward")
    plt.legend(borderpad=0.3,
               loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_yaxis().tick_left()
    ax.get_xaxis().tick_bottom()
    plt.xlim([0, 5500])
    plt.ylim([0, 250])

    plt.savefig(r"D:\Documents\officesvn\writing\cogsci2014\figures\pretraining.png")
    plt.savefig(r"D:\Documents\officesvn\writing\cogsci2014\figures\pretraining.pdf")


hierarchical_vs_flat()
pretrained_vs_nopretrained()

plt.show()
