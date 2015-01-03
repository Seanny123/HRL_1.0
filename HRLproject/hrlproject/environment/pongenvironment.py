import os
import subprocess
import math

from hrlproject.environment.environmenttemplate import EnvironmentTemplate
from hrlproject.misc.pipereader import PipeReader
from hrlproject.misc.HRLutils import rand as random

class PongEnvironment(EnvironmentTemplate):
    def __init__(self, actions, playernum=0):
        EnvironmentTemplate.__init__(self, "PongEnvironment", 2, actions)


        self.state_radius = 0.9
        self.max_y = 480
        self.playernum = playernum
        self.place_dev = 0.2
        self.rewardscale = 5
        self.optimal_move = 0
        self.stats = [0]*4
        self.mapping = {"up":-1, "stay":0, "down":1}
        
        self.placecells = self.gen_placecells(min_spread=self.place_dev * 0.5)
        self.place_activations = [0 for _ in self.placecells]

        self.reader = None
        self.error = None

        self.create_origin("place", lambda: self.place_activations)
        self.create_origin("stats", lambda: self.stats)
        
    def start_pong(self):
        # start up the pong game subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = r"C:\Python27"
        self.p = subprocess.Popen("python -u pong.py",
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        cwd=os.path.join(os.path.dirname(__file__), "..", "misc"),
                        env=env)
        print "subprocess started"
        
        # thread to read from subprocess output
        self.reader = PipeReader(self.p.stdout)
        self.error = PipeReader(self.p.stderr)

    def tick(self):
        self.reward *= 0.97 # note: doing a slow decay rather than setting to 0, to put
                            # a bit more power in the reward signal
        
        data = self.reader.readline()
        while data is not None:
            data = data.split()
            if data[0] == "state":
                self.state = [float(data[2]), float(data[3 + self.playernum])]
                self.state = [2 * self.state_radius * x / self.max_y - self.state_radius for x in self.state]
                self.optimal_move = 1 if float(data[2]) > float(data[4-self.playernum]) else -1
            elif data[0] == "reward" and int(data[1]) == self.playernum:
                self.reward = float(data[2]) * self.rewardscale
            elif data[0] == "stats":
                self.stats = [float(x) for x in data[1:]]
#            else:
#                print "unrecognized input (%s)" % data

            data = self.reader.readline()
            
        self.place_activations = self.calc_activations(self.state, self.place_dev)

#        print "state:", self.state
        print >> self.p.stdin, "move %d %f" % (1 - self.playernum, random.choice([self.optimal_move*1,1,-1]))
        print >> self.p.stdin, "move %d %f" % (self.playernum, self.mapping[self.action[0]])
        
    def calc_activations(self, loc, place_dev):
#        dists = np.sqrt(np.sum(((self.placecells - np.array(loc)) ** 2), axis=1))
#        dists_old = np.asarray([self.calc_dist(p, loc) for p in self.placecells])
#        print "dists", dists
#        print "dists_old", dists_old
        dists = [self.calc_dist(p, loc) for p in self.placecells]
        return [math.exp(-d ** 2 / (2 * place_dev ** 2)) for d in dists]

    def gen_placecells(self, min_spread=0.2):
        """Generate the place cell locations that will give rise to the state representation.

        :param min_spread: the minimum distance between place cells
        """

        N = None
        num_tries = 1000 # a limit on the number of attempts to place a new placecell

        # assign random x,y locations to each neuron
        locations = [self.random_location()]
        while True:
            # generate a random new point
            new_loc = self.random_location()

            # check that the point isn't too close to previous points
            count = 0
            while min([self.calc_dist(new_loc, l) for l in locations]) < min_spread and count < num_tries:
                new_loc = self.random_location()
                count += 1

            # add the new point
            locations += [new_loc]

            if (N == None and count >= num_tries) or len(locations) == N:
                # stop when required number of place cells built (if N specified),
                # or when world has been decently filled
                break

        return locations

    def random_location(self, radius=1):
        return (random.random() * 2 * radius - radius,
                random.random() * 2 * radius - radius)

    def gen_encoders(self, N):
        """Generate encoders for state population in RL agent."""

        locs = self.placecells

        encoders = [None for _ in range(N)]
        for i in range(N):
            # pick a random point for the neuron
            pt = self.random_location() # could make this avoid walls if we want

            # set the encoder to be the inverse of the distance from each placecell to that point
            encoders[i] = [1.0 / self.calc_dist(pt, l) for l in locs]

            # cut off any values below a certain threshold
            encoders[i] = [x if x > 0.5 * max(encoders[i]) else 0 for x in encoders[i]]

            # normalize the encoder
            encoders[i] = [x / math.sqrt(sum([y ** 2 for y in encoders[i]])) for x in encoders[i]]

        return encoders

    def calc_dist(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
