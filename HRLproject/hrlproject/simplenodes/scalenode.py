from ca.nengo.math.impl import GaussianPDF

import nef
import pdb

from hrlproject.misc import HRLutils

class TimeScale(nef.SimpleNode):
    """Scale decreases with time. Pass in a linear, exponential or whatever function
    exclusively time-dependent function
    
    :output scale: scalar amount to scale the noisy values
    """
    
    def __init__(self, frequency, dimension=1, scale_func=None, label=""):
        """Initialize node variables.
        
        :param frequency: frequency to update noise values
        :param dimension: dimension of noise signal
        """
        
        self.period = 1.0/frequency
        self.updatetime = 0.0
        self.scale_func = scale_func
        self.state = [0.0 for _ in range(dimension)]
        self.pdf = GaussianPDF(0,1)
        
        nef.SimpleNode.__init__(self, "TimeScaleNode_"+label)
        
    def tick(self):
        if self.t > self.updatetime:
            #print("scale %s, %s" %(self.t, self.scale_func(self.t)))
            self.state = [self.pdf.sample()[0]*self.scale_func(self.t) for _ in range(len(self.state))]
            self.updatetime = self.t + self.period
        
    def origin_state(self):
        return self.state

    def origin_scale(self):
        """for debugging only"""
        return [self.scale_func(self.t)]


class StateScale(nef.SimpleNode):
    """Based on the HIGHEST state frequency pass a scale
    we ignore the Q values because we want to encourage constant exploration
    Basically, we make an agent who will never be satisfied
    It will be interesting to see how often this thing oscillates
    Or should I drive the agent towards unexplored states?
    But putting it in a node is going to cause LAG. Which is fine because it's 
    not like we need super instantaneous results here.
    This is kind of silly with a discrete space that it's not hard to completely
    explore, but in a large continuous space the idea of moving "towards" makes
    a lot more sense
    
    :input scale: scale on the output values (modifying standard deviation from base of 1)
    :output scale: vector to scale the noisy values
    """
    
    def __init__(self, frequency, action_dimension=1, grid_dimension=2, grid_height=3, grid_width=3, time_constant=0.003, distance_constant=0.01875):
        """Initialize node variables.
        
        :param frequency: frequency to update noise values
        :param dimension: dimension of noise signal
        """
        
        self.time_constant = time_constant
        self.distance_constant = distance_constant
        self.period = 1.0/frequency
        self.updatetime = 0.0
        self.grid_dimension = grid_dimension
        self.action_dimension = action_dimension
        self.agent_state = [0.0 for _ in range(grid_dimension)]
        self.state = [0.0 for _ in range(action_dimension)]
        self.scale = [0.0 for _ in range(action_dimension)]
        # assuming two dimensional state, because I'm lazy
        self.state_visited = [[0 for _ in range(grid_width)] for _ in range(grid_height)]
        self.pdf = GaussianPDF(0,1)

        self.xoffset = grid_width / 2
        self.yoffset = grid_height / 2
        nef.SimpleNode.__init__(self, "StateScaleNode")

        #self.create_termination("state", termination_state)

    def tick(self):
        if self.t > self.updatetime:
            self.scale = [0.0 for _ in range(self.action_dimension)]
            # the least visited vector could also be found by checking all the tiles and weighing them by the last visit time, instead of just looking for the minimum 
            min_list = []
            min_val = self.state_visited[0][0]
            # get min directions from the data structure
            # this is basically the gradient descent problem?
            # only if the dataset is too big to just iterate through?
            # find the global minimums O(n)
            for i in range(len(self.state_visited)):
                for j in range(len(self.state_visited[i])):
                    if(self.state_visited[i][j] == min_val):
                        min_list.append([i, j])
                    elif(self.state_visited[i][j] < min_val):
                        min_list = []
                        min_val = self.state_visited[i][j]
                        min_list.append([i, j])
            # take the average of their orientation # runtime: O(n)
            # by taking the average of the list of minimum vectors
            total = [0.0 for _ in range(self.grid_dimension)]
            for val in min_list:
                total[0] += val[0] - self.xoffset
                total[1] += val[1] - self.yoffset
            least_visited = [total[0] / len(total), total[1] / len(total)]
            # convert the average minimum vector to a scale
            # because actions are encoded to up, right, down, left
            # set the scale proportional to the time it was last visited versus the current time
            closest_min = self.agent_state
            min_state_dist = HRLutils.distance(self.agent_state, min_list[0])
            for min_loc in range(len(min_list)):
                state_dist = HRLutils.distance(self.agent_state, min_list[min_loc])
                if(state_dist < min_state_dist):
                    closest_min = min_list[min_loc]
                    min_state_dist = state_dist

            least_visited[0] += self.xoffset
            least_visited[1] += self.yoffset
            state_diff = HRLutils.difference(self.agent_state, least_visited)
            # Whats the point of state_diff? # It catches the corner case where the least visited node is the one you're already on, which may or may not be a real thing that happens

            # To summarize this noise boost operation, all it's accomplishing is discouraging the agent to go to really far places or to a place that it's visited recently
            hor_min_dist = abs(self.agent_state[0] - closest_min[0])
            hor_noise_boost = (
                (self.t - min_val) * self.time_constant
                + (1/(1+hor_min_dist)) * self.distance_constant
            ) * (state_diff[0] != 0)

            vert_min_dist = abs(self.agent_state[1] - closest_min[1])
            vert_noise_boost = (
                (self.t - min_val) * self.time_constant
                + (1/(1+vert_min_dist)) * self.distance_constant
            ) * (state_diff[1] != 0)

            # this is a bit of a clustermuffin for mapping
            if(state_diff[1] > 0):
                # go left
                print("boost left")
                self.scale[3] = hor_noise_boost
            elif(state_diff[1] < 0):
                # got right
                print("boost right")
                self.scale[1] = hor_noise_boost

            if(state_diff[0] < 0):
                # got down
                print("boost down")
                self.scale[2] = vert_noise_boost
            elif(state_diff[0] > 0):
                # got up
                print("boost up")
                self.scale[0] = vert_noise_boost

            print("Current state %s" %self.agent_state)
            print("least_visited %s" %least_visited)
            print("state_diff %s" %state_diff)
            print("scale: %s" %self.scale)
            #pdb.set_trace()
            self.state = [self.pdf.sample()[0]*self.scale[i] for i in range(len(self.state))]
            self.updatetime = self.t + self.period

    def origin_state(self):
        return self.state

    def origin_scale(self):
        """for debugging only"""
        return self.scale


    def termination_agent_state(self, x, dimensions=2):
        #print("Current state %s" %x)
        self.agent_state = x
        #print(self.state_visited)
        # keep track of state # should probably wrap this up in an update or something
        self.state_visited[ int(x[0]) ][ int(x[1]) ] = self.t


class ErrorScale(nef.SimpleNode):
    """Node to output gaussian noise with mean 0 and standard deviation driven
    by an input signal.
    
    :input scale: scale on the output values (modifying standard deviation from base of 1)
    :output noise: vector of noisy values
    """
    
    def __init__(self, frequency, dimension=1, constant=2):
        """Initialize node variables.
        
        :param frequency: frequency to update noise values
        :param dimension: dimension of noise signal
        """
        
        self.period = 1.0/frequency
        self.updatetime = 0.0
        self.state = [0.0 for _ in range(dimension)]
        self.scale = 0.03
        self.error = 0.0
        self.constant = constant
        self.pdf = GaussianPDF(0,1)
        nef.SimpleNode.__init__(self, "ErrorScaleNode")
        
    def tick(self):
        if self.t > self.updatetime:
            self.state = [self.pdf.sample()[0]*self.scale for _ in range(len(self.state))]
            self.updatetime = self.t + self.period
    
    # can I make this method in init without anything weird happening?
    def termination_error(self, x, dimensions=4):
        """get the error from the error node and see if it's increasing or not"""
        self.error = 0
        for val in x:
            self.error += abs(val)
        self.scale = self.constant * self.error

    def origin_state(self):
        return self.state

    def origin_scale(self):
        "for debugging only"
        return [self.scale]

# There's a part of me that wants to implement combined error and state
# So it goes to unexplored places only when the error is large
# So letting the error tune the time constant