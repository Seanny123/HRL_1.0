import os

from hrlproject.misc import HRLutils


def combine_files():
    path = os.path.join("..", "..", "data", "delivery", "flat", "dataoutput_2")

    data = []
    for i in range(10):
        try:
            data += [HRLutils.load_data(path + ".%s.txt" % i)]
        except IOError:
            continue

    print "found %s files to combine" % len(data)
    print len(data[0]), "records"

    starttime = 0.0
    newdata = [[] for _ in data[0]]
    for d in data:
        if len(d) != len(newdata):
            print "uh oh, number of records is wrong"
            print len(d), len(newdata)
        for i, record in enumerate(d):
            for entry in record:
                newdata[i] += [[entry[0] + starttime, entry[1]]]
        starttime = newdata[0][-1][0]

    HRLutils.save_data(path + "_combined.txt", newdata)

combine_files()
