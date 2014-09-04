import nef

from ca.nengo.plot.impl import DefaultPlotter


class DataNode(nef.SimpleNode):
    """Node to collect data and output it to file."""

    def __init__(self, show_plots=None, period=1, filename=None):
        """Initialize node variables.

        :param show_plots: if True, display graphs as the node is running
        :param period: specifies how often the node should create a new data entry
        :param filename: name of file to save data to
        """

        nef.SimpleNode.__init__(self, "DataNode")

        self.show_plots = show_plots
        self.period = period
        self.filename = filename

        self.sources = []
        self.records = []
        self.filters = []
        self.types = []

        self.plotter = DefaultPlotter()

    def record(self, origin, filter=1.0, func=lambda x: x):
        """Record data from the given origin.

        :param origin: origin to record data from
        :param filter: time constant for exponential filter applied to data
        :param func: function applied to the output of origin
        """

        output = func([0.0]*origin.getDimensions())
        if isinstance(output, (float, int)):
            output = [output]
            
        self.filters += [filter]
        self.sources += [origin]
        self.records += [[[self.t, [0.0]*len(output)]]]
        self.types += [func]

    def record_avg(self, origin, filter=0.0):
        self.record(origin, filter, lambda s: [float(sum(s)) / len(s)])

    def record_sparsity(self, origin, filter=0.0):
        self.record(origin, filter, lambda s: [len([x for x in s if x == 0.0]) / float(len(s))])

    def tick(self):
        for i, r in enumerate(self.records):
            f = self.filters[i]

            # get data from origin
            try:
                s = self.sources[i].getValues().getValues()
            except:
                # this can fail if the simulator is currently in the process
                # of writing to the origin
                continue

            # apply function to data
            s = self.types[i](s)
            if isinstance(s, (float, int)):
                s = [s]

            # apply filter to data
#            r[-1][1] *= 1 - f
#            r[-1][1] += f * s
            r[-1][1] = [x*(1-f) + y*f for x,y in zip(r[-1][1], s)]

            # if period has elapsed, create a new data entry
            if self.t % self.period < 1e-5:
                r += [[self.t, r[-1][1]]]

                # write data to file
                if self.filename != None:
                    f = open(self.filename, "w")
                    f.write("\n".join([";".join([" ".join([str(v) for v in [entry[0]]+entry[1]]) for entry in r]) for r in self.records]))
                    f.close()

        if self.show_plots != None and self.t % self.show_plots < 1e-5:
            for i, r in enumerate(self.records):
                self.plotter.doPlot([x[0] for x in r], [x[1] for x in r], str(i))