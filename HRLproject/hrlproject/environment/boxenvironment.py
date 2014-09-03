import random, copy

from hrlproject.environment.environmenttemplate import EnvironmentTemplate

class BoxEnvironment(EnvironmentTemplate):
    """A continuous environment represented by a set of boxes defining different regions.

    :input action: vector representing action selected by agent
    :output state: current x,y location of agent
    :output reward: reward value
    :output learn: 1 if the agent should be learning
    :output reset: 1 if the agent should reset its error calculation
    """

    def __init__(self, stateD, actions, filename, name="BoxWorld", delay=0.1, cellwidth=1.0, dx=0.01, cartesian=False):
        """Initializes environment variables.

        :param stateD: dimension of state
        :param actions: actions available to the system
            :type actions: list of tuples (action_name,action_vector)
        :param filename: name of file containing map description
        :param name: name for environment
        :param delay: time to wait between action updates
        :param cellwidth: physical distance represented by each character in map file
        :param dx: distance agent moves in one timestep
        :param cartesian: if True, represent the agent's location in x,y cartesian space (0,0 in centre)
            if False, agent's location is in matrix space (0,0 in top left)
        """

        EnvironmentTemplate.__init__(self, name, stateD, actions)

        self.wallboxes = []
        self.targetboxes = []
        self.mudboxes = []
        self.worldbox = None

        self.delay = delay
        self.update_time = 0.5 # the time to perform the next action update
        self.learntime = [-1, -1]
        self.resettime = [0.05, 0.1] # reset right at the beginning to set things up
        self.rewardresettime = 0.6 #time agent will spend in reward location before being reset
        self.num_actions = len(actions)
        self.chosen_action = None
        self.cellwidth = cellwidth
        self.dx = dx

        f = open(filename)
        data = [line[:-1] if line.endswith("\n") else line for line in f]
        f.close()

        if cartesian:
            # modify all the coordinates so that they lie in the standard cartesian space
            # rather than the matrix row/column numbering system
            self.yoffset = (len(data) / 2) * self.cellwidth
            self.yscale = -1
            self.xoffset = (len(data[0]) / 2) * self.cellwidth
        else:
            self.yoffset = 0
            self.yscale = 1
            self.xoffset = 0

        self.load_boxes(data)

        self.i_state = self.random_location()
            # internal state is the location in terms of the internal coordinate system (i.e. 0,0 in the top left)
        self.state = self.transform_point(self.i_state)
            # this is the state in terms of the agent's coordinate system (usually cartesian where 0,0 is the middle)

        self.create_origin("learn", lambda: [1.0 if self.t > self.learntime[0] and self.t < self.learntime[1] else 0.0])
        self.create_origin("reset", lambda: [1.0 if self.t > self.resettime[0] and self.t < self.resettime[1] else 0.0])

    def tick(self):
        # update action during reset period
        if self.t > self.resettime[0] and self.t < self.resettime[1]:
            self.chosen_action = copy.deepcopy(self.action)

        # update state
        if self.chosen_action != None:
            if self.chosen_action[0] == "up":
                dest = [self.i_state[0], self.i_state[1] - self.dx]
            elif self.chosen_action[0] == "right":
                dest = [self.i_state[0] + self.dx, self.i_state[1]]
            elif self.chosen_action[0] == "down":
                dest = [self.i_state[0], self.i_state[1] + self.dx]
            elif self.chosen_action[0] == "left":
                dest = [self.i_state[0] - self.dx, self.i_state[1]]
            else:
                print "Unrecognized action"

            if not self.is_wall(dest):
                self.i_state = dest
                self.state = self.transform_point(self.i_state)

        #update reward
        if self.is_target(self.i_state):
            self.reward = 1

            if self.rewardstart == None:
                self.rewardstart = self.t
            else:
                # check if we've been in the reward for long enough to reset
                if self.t - self.rewardstart > self.rewardresettime:
                    self.i_state = self.random_location()
                    self.state = self.transform_point(self.i_state)
        else:
            self.reward = 0
            self.rewardstart = None

        #update learning/reset times
        if self.t > self.update_time:

            if isinstance(self.delay, float):
                self.update_time = self.t + self.delay
            else:
                self.update_time = self.t + random.uniform(self.delay[0], self.delay[1])

            # calculate learn/reset periods
            statedelay = 0.2 # time to wait after update triggers
            learninterval = 0.1 # time to learn for
            resetdelay = 0.1 # time between learn and reset
            resetinterval = 0.05 # time to reset for

            self.learntime = [self.t + statedelay, self.t + statedelay + learninterval]
            self.resettime = [self.learntime[1] + resetdelay, self.learntime[1] + resetdelay + resetinterval]

    def transform_point(self, pt):
        """Transform a point from internal space (0,0 top left) to agent's coordinate
        system (usually Cartesian)."""

        x, y = pt
        return (x - self.xoffset, (y - self.yoffset) * self.yscale)

    def random_location(self):
        """Pick a random location that isn't in a wall or target."""

        while True:
            pt = (random.uniform(self.worldbox.tl[0], self.worldbox.br[0]),
                  random.uniform(self.worldbox.tl[1], self.worldbox.br[1]))
            if not self.is_wall(pt) and not self.is_target(pt):
                return pt

    def load_boxes(self, data):
        """Translate map data into a set of Boxes."""

        # worldbox represents the total map area
        self.worldbox = self.Box((0, 0), (len(data[0]) * self.cellwidth, len(data) * self.cellwidth))

        # create a box corresponding to each character/cell in the map file
        tl_x = 0
        tl_y = 0
        for row in data:
            for cell in row:
                if cell == ".":
                    self.wallboxes += [self.Box((tl_x, tl_y), (tl_x + self.cellwidth, tl_y + self.cellwidth))]
                elif cell == "x":
                    self.targetboxes += [self.Box((tl_x, tl_y), (tl_x + self.cellwidth, tl_y + self.cellwidth))]
                tl_x += self.cellwidth
            tl_x = 0
            tl_y += self.cellwidth

    def get_boxes(self):
        """Return a list of boxes (used for interactive mode display)."""

        boxes = [(" ", self.worldbox.tl, self.worldbox.br)]
#        boxes = []
        boxes += [(".", b.tl, b.br) for b in self.wallboxes]
        boxes += [("x", b.tl, b.br) for b in self.targetboxes]
        agentscale = 100
        boxes += [("a", (self.i_state[0] - self.dx * agentscale, self.i_state[1] - self.dx * agentscale),
                   (self.i_state[0] + self.dx * agentscale, self.i_state[1] + self.dx * agentscale))]
        return boxes

    def is_wall(self, pt):
        return len([x for x in self.wallboxes if x.contains(pt)]) > 0

    def is_target(self, pt):
        return len([x for x in self.targetboxes if x.contains(pt)]) > 0

    class Box:
        def __init__(self, topleft, bottomright):
            self.tl = topleft
            self.br = bottomright

        def contains(self, pt):
            x, y = pt
            return x >= self.tl[0] and x <= self.br[0] and \
                y >= self.tl[1] and y <= self.br[1]
