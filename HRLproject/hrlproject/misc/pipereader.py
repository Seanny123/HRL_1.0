"""Based on http://eyalarubas.com/python-subproc-nonblock.html"""

from threading import Thread
import Queue

class PipeReader(Thread):
    def __init__(self, pipe):
        Thread.__init__(self)

        self.pipe = pipe
        self.queue = Queue.Queue()
        self.daemon = True

#        self.started = False
        self.start()

    def run(self):
        while True:
            data = self.pipe.readline()
            if data:
                self.queue.put(data)
            else:
                print "pipe closed"
                break

    def readline(self):
#        if not self.started:
#            print "starting thread"
#            self.start()
#            self.started = True

        if not self.queue.empty():
            return self.queue.get(block=False)
        return None
    
    def readlines(self):
        lines = []
        tmp = self.readline()
        while tmp is not None:
            lines += [tmp]
            tmp = self.readline()
        return lines

    def readlatest(self):
        while not self.queue.empty():
            data = self.queue.get(block=False)
        return data
