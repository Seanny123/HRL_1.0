import nef

from ca.nengo.util import MU

from hrlproject.misc import HRLutils

class BadrePseudoreward(nef.SimpleNode):
    """Node used to generate pseudoreward for lower level in the Badre task.

    :input action: current action selected by low level
    :input context: current context selected by high level
        (indicating which aspect of stimuli should be driving action selection)
    :output pseudoreward: pseudoreward value for low level
    """

    def __init__(self, env, contexts):
        """Initializes node variables.

        :param env: BadreEnvironment for this task
        :param contexts: list of (name, vector) tuples for possible contexts
        """

        nef.SimpleNode.__init__(self, "BadrePseudoreward")

        self.env = env
        self.contexts = contexts
        self.action = None
        self.context = None
        self.reward = 0
        self.rewardval = 1.5

        self.getTermination("context").setDimensions(len(self.contexts[0][1]))
        self.create_origin("pseudoreward", lambda : [self.reward])

    def tick(self):
        # check if env is currently giving reward (we want to give pseudoreward at the same time)
        if self.env.reward != 0:
            if self.target_answer is None:
                self.reward = 0
            else:
                # check if the selected action matches the correct action
                self.reward = self.rewardval if HRLutils.similarity(self.target_answer, self.action) > 0.5 else -self.rewardval
        else:
            self.reward = 0

            # update the target_answer (the action the low level should be selecting given
            # the current context)
            if self.context[0] == "orientation":
                self.target_answer = self.env.state[:self.env.num_orientations]
            elif self.context[0] == "shape":
                self.target_answer = self.env.state[self.env.num_orientations:-self.env.num_colours]
            else:
                self.target_answer = None

    def termination_action(self, a, pstc=0.01, dimensions=3):
        self.action = HRLutils.normalize(a)

    def termination_context(self, c, pstc=0.01):
        self.context = max(self.contexts, key=lambda x: MU.prod(HRLutils.normalize(c), HRLutils.normalize(x[1])))
