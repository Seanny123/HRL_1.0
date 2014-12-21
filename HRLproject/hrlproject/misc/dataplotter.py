"""Simple script for plotting the output of datanode."""

import os

import pylab as plt

filename = os.path.join("..", "..", "data", "dataoutput_pretrain.txt")

#plt.ion()

lines = []
axes = []

#while True:
# read in data from file
f = open(filename)
data = [[[float(v) for v in entry.split(" ")] for entry in r.split(";")] for r in f.readlines()]
f.close()

# add an extra data entry that is the total accumulated reward (assuming
# reward is in data[0]).  kind of hacky, but whatever
rewardsum = [[0, 0]]
for i, x in enumerate(data[0]):
    rewardsum += [[x[0], rewardsum[i - 1][1] + x[1]]]
data += [rewardsum]

for i, d in enumerate(data):
    x = [x[0] for x in d]
    y = [y[1:] for y in d]

    # create the figure if it hasn't already been created
    if i >= len(lines):
        fig = plt.figure(i)
#        ax = fig.add_subplot(111)
#        axes += [ax]
#        lines += [ax.plot([0], [0]*len(y[0]))[0]]

    # update axes
    miny = min([min(v) for v in y])
    maxy = max([max(v) for v in y])
    plt.xlim([min(x), max(x)])
    plt.ylim([miny * (0.9 if miny > 0 else 1.1), maxy * 1.1])
#    axes[i].set_xlim(min(x), max(x))
#    axes[i].set_ylim(miny * (0.9 if miny > 0 else 1.1), maxy * 1.1)

    # update data
    plt.plot(x, y)
#    lines[i].set_data(x, y)

plt.show()

#    plt.draw()
#    plt.pause(10)
