import random
import copy
import pdb

from hrlproject.environment import environmenttemplate as et

class GridWorldEnvironment(et.EnvironmentTemplate):
    """Spatial environment based on a discrete set of grid locations.

    :input action: vector representing action selected by agent
    :input Qs: Q values from agent (just used for display)
    :output state: vector representing current state
    :output reward: reward value
    :output learn: 1 if the agent should be learning
    :output reset: 1 if the agent should reset its error calculation
    """

    def __init__(self, stateD, actions, filename, name="GridWorld", cartesian=False, delay=0.1, datacollection=False, default_reward = 0.0, state_filename=""):
        """Initializes environment variables.

        :param stateD: dimension of state
        :param actions: actions available to the system
            :type actions: list of tuples (action_name,action_vector)
        :param filename: name of file containing map description
        :param name: name for environment
        :param cartesian: if True, represent the agent's location in x,y cartesian space (0,0 in centre)
            if False, agent's location is in matrix space (0,0 in top left)
        :param delay: time to wait between action updates
        :param datacollection: if True, agent moves in a fixed pattern through the states
        """

        et.EnvironmentTemplate.__init__(self, name, stateD, actions)

        self.state_filename = state_filename
        if(state_filename != ""):
            f = open(state_filename, "w")
            f.close()

        self.cartesian = cartesian
        self.delay = delay # time to spend in each grid state (can be a float, or tuple specifying range)
        self.update_time = 0.5
        self.learntime = [-1, -1]
        self.resettime = [0.05, 0.1] # reset right at the beginning to set things up
        self.Qs = {} # store the Q values, just for display #WTF is the format of these things?
        self.num_actions = len(actions)
        self.chosen_action = None
        self.datacollection = datacollection

        #data collection
        self.latencies = [] # record of how long it took (relative to optimal) to reach goal on each trial
        self.stepcount = 0 # number of steps in the current trial
        self.manhat = 0 # optimal number of steps for the current trial
        self.cellcount = 0 # controls movement pattern if datacollection=True


        f = open(filename)
        data = f.readlines()
        f.close()

        for i, line in enumerate(data):
            if line.endswith("\n"):
                data[i] = line[:-1]

        if cartesian:
            # modify all the coordinates so that they lie in the standard cartesian space
            # rather than the matrix row/column numbering system
            self.yoffset = len(data) / 2
            self.yscale = -1
            self.xoffset = len(data[0]) / 2
        else:
            self.yoffset = 0
            self.yscale = 1
            self.xoffset = 0

        # Create a grid object where the dots are barriers, the spaces are space and the x is the goal
        self.grid = [[self.Cell(c, j - self.xoffset, self.yscale * (i - self.yoffset)) for j, c in enumerate(row)] for i, row in enumerate(data)]
        self.state = self.pickRandomLocation()

        self.create_origin(
            "learn",
            lambda: [1.0 if self.t > self.learntime[0] and self.t < self.learntime[1] else 0.0]
        )
        self.create_origin(
            "reset", 
            lambda: [1.0 if self.t > self.resettime[0] and self.t < self.resettime[1] else 0.0]
        )

        def store_Qs(x, dimensions=len(actions), pstc=0.01):
            self.Qinput = x
        self.create_termination("Qs", store_Qs)

    def tick(self):
        """The basic cycle of the environment is act, learn and reset. This method takes care of that scheduling"""
        # check if we want to do a state update
        if self.t > self.update_time:

            if isinstance(self.delay, float):
                self.update_time = self.t + self.delay
            else:
                self.update_time = self.t + random.uniform(self.delay[0], self.delay[1])

            self.stepcount += 1

            # store the current Qval inputs
            x, y = self.state
            # (x, y) = all_the_Qs
            self.Qs[(int(x + self.xoffset), int(self.yscale * y + self.yoffset))] = [n for n in self.Qinput]

            # update state
            if self.getCell(self.state).target:

                if(self.state_filename != ""):
                    file_writer = open(self.state_filename, "a")
                    file_writer.write("r\n")
                    file_writer.close()

                print("REACHED THE TARGET")
                self.state = self.pickRandomLocation()

                # data collection
                self.latencies += [self.stepcount - self.manhat - 1]
                    # latency for just completed trial

                    # reset for next trial
                self.stepcount = 0
                agentloc = self.getCell(self.state)
                targetloc = [c for row in self.grid for c in row if c.target][0]
                self.manhat = abs(agentloc.x - targetloc.x) + abs(agentloc.y - targetloc.y)

#                print self.latencies
#                self.plots()
            elif self.chosen_action != None:
                if self.chosen_action[0] == "up":
                    dest = self.getCell([self.state[0], self.state[1] - 1 * self.yscale])
                elif self.chosen_action[0] == "right":
                    dest = self.getCell([self.state[0] + 1, self.state[1]])
                elif self.chosen_action[0] == "down":
                    dest = self.getCell([self.state[0], self.state[1] + 1 * self.yscale])
                elif self.chosen_action[0] == "left":
                    dest = self.getCell([self.state[0] - 1, self.state[1]])
                else:
                    print "Unrecognized action"
                print("Going %s" %self.chosen_action[0])

                if not dest.wall:
                    self.state = dest.location()

            if(self.state_filename != ""):
                file_writer = open(self.state_filename, "a")
                file_writer.write("%s %s %s" %(self.t, self.state[0], self.state[1]))
                file_writer.write("\n")
                file_writer.close()
            print("%s STATE: %s" %(self.t, self.state))

            # add extra time in this state if it's mud
            if self.getCell(self.state).mud:
                self.update_time += 3.0

            # update reward
            if self.getCell(self.state).target:
                print("HOT DAMN REWARD")
                self.reward = 1
            else:
                # as seen in delivery task, negative reward for lally-gagging
                # temporarily zero because testing
                self.reward =  0.0

            # calculate the new learn/reset periods
            statedelay = 0.2 # time to wait after a statechange
            learninterval = 0.1 # time to learn for
            resetdelay = 0.1 # time between learn and reset
            resetinterval = 0.05 # time to reset for

            self.learntime = [
                self.t + statedelay, 
                self.t + statedelay + learninterval
            ]
            # obviously the reset time
            self.resettime = [
                self.learntime[1] + resetdelay, 
                self.learntime[1] + resetdelay + resetinterval
            ]

            # override movement for data collection
            if self.datacollection:
                gridsize = 5
                self.state = self.grid[self.cellcount % gridsize + 1][(self.cellcount / gridsize) % gridsize + 1].location()
                self.cellcount += 1

        # check if we want to look for an action from the agent
        if self.t > self.resettime[0] and self.t < self.resettime[1]:
            # Get the action from agent input
            self.chosen_action = copy.deepcopy(self.action)

    def getCell(self, location):
        """Translate x,y location (usually in Cartesian space) into matrix cell."""

        x, y = location
        cell = self.grid[int(self.yscale * y + self.yoffset)][int(x + self.xoffset)]

        return cell

    def pickRandomLocation(self):
        while True:
            cell = random.choice(random.choice(self.grid))
            if not cell.wall and not cell.target:
                return cell.location()

    def __str__(self):
        result = ""
        for row in self.grid:
            for cell in row:
                if cell.location() == self.state:
                    result += "a"
                else:
                    result += cell.data
            result += "\n"
        return result

    def getQs(self):
        return self.Qs

    class Cell:
        def __init__(self, data, x, y):
            self.data = data
            self.x = x
            self.y = y

            # note: x,y represents the agent's location in the abstract physical space.
            # each cell is in an i,j location in the grid (getCell above maps between the two)

            self.wall = False
            self.mud = False
            self.target = False

            if data == ".":
                self.wall = True
            elif data == "_":
                self.mud = True
            elif data == "X" or data == "x":
                self.target = True

        def location(self):
            return [self.x, self.y]

        def __repr__(self):
#            return "[%s : (%3d,%3d)]" % (self.data, self.x, self.y)
            return self.data
