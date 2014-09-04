from __future__ import with_statement

"""Functions for running key tasks on the model."""

import os
import socket

import nef
import sys

#Based off the hostname, set the apprpriate path
#compname = os.environ["COMPUTERNAME"] 
compname = socket.gethostname()
if compname == "CTN04" or compname == "DANIEL-PC":
    sys.path.append(r"D:\Documents\officesvn\HRLproject")
elif compname == "ctngpu2":
    sys.path.append("/home/ctnuser/drasmuss/HRLproject")
elif compname == "hybrid":
    sys.path.append("/home/sean/HRL_link")
elif compname == "ctn11":
    sys.path.append("/home/saubin/Github/HRL_1.0")
    sys.path.append("/home/saubin/Github/HRL_1.0/HRLproject/hrlproject")
    sys.path.append("/home/saubin/Github/HRL_1.0/HRLproject")
else:  # assume running on sharcnet
    compname = "sharcnet"
    sys.path.append("/home/drasmuss/HRLproject")

from ca.nengo.util import MU
from ca.nengo.util.impl import NodeThreadPool
from ca.nengo.model import SimulationMode

from hrlproject.environment import deliveryenvironment, contextenvironment, badreenvironment
from hrlproject.misc import HRLutils
from hrlproject.agent import smdpagent
from hrlproject.simplenodes import terminationnode, datanode

# What's the difference between the nav_agent and the ctrl_agent? They're the two levels of the hierarchy
def run_deliveryenvironment(navargs, ctrlargs, tag=None, seed=None):
    """Runs the model on the delivery task.

    :param navargs: kwargs for the nav_agent (see SMDPAgent.__init__)
    :param ctrlargs: kwargs for the ctrl_agent (see SMDPAgent.__init__)
    :param tag: string appended to datafiles associated with this run
    :param seed: random seed used for this run
    """

    if seed is not None:
        HRLutils.set_seed(seed)
    seed = HRLutils.SEED

    if tag is None:
        tag = str(seed)

    net = nef.Network("runDeliveryEnvironment", seed=seed)

    stateN = 1200 # number of neurons to use in state population
    contextD = 2 # dimension of context vector
    context_scale = 1.0 # relative scale of context vector vs state vector
    max_state_input = 2 # maximum length of the input vector to the state population
    actions = [("up", [0, 1]), ("right", [1, 0]), ("down", [0, -1]), ("left", [-1, 0])]
        # labels and vectors corresponding to basic actions available to the system

    if navargs.has_key("load_weights") and navargs["load_weights"] is not None:
        navargs["load_weights"] += "_%s" % tag
    if ctrlargs.has_key("load_weights") and ctrlargs["load_weights"] is not None:
        ctrlargs["load_weights"] += "_%s" % tag

    ###ENVIRONMENT

    env = deliveryenvironment.DeliveryEnvironment(actions, HRLutils.datafile("contextmap.bmp"),
                                                  colormap={-16777216:"wall",
                                                           - 1:"floor",
                                                           - 256:"a",
                                                           - 2088896:"b"},
                                                  imgsize=(5, 5), dx=0.001, placedev=0.5)
    net.add(env)

    print "generated", len(env.placecells), "placecells"



    ###NAV AGENT

    # generate encoders and divide them by max_state_input (so that inputs
    # will be scaled down to radius 1)
    enc = env.gen_encoders(stateN, contextD, context_scale)
    enc = MU.prod(enc, 1.0 / max_state_input)

    # read in eval points from file
    with open(HRLutils.datafile("contextbmp_evalpoints_%s.txt" % tag)) as f:
        evals = [[float(x) for x in l.split(" ")] for l in f.readlines()]

    nav_agent = smdpagent.SMDPAgent(stateN, len(env.placecells) + contextD, actions, name="NavAgent",
                                    state_encoders=enc, state_evals=evals, state_threshold=0.8,
                                    **navargs)
    net.add(nav_agent)

    print "agent neurons:", nav_agent.countNeurons()

    # output of nav_agent is what goes to the environment
    net.connect(nav_agent.getOrigin("action_output"), env.getTermination("action"))

    # termination node for nav_agent (just a timer that goes off regularly)
    nav_term_node = terminationnode.TerminationNode({terminationnode.Timer((0.6, 0.9)):None}, env,
                                                    contextD=2, name="NavTermNode")
    net.add(nav_term_node)

    net.connect(nav_term_node.getOrigin("reset"), nav_agent.getTermination("reset"))
    net.connect(nav_term_node.getOrigin("learn"), nav_agent.getTermination("learn"))
    net.connect(nav_term_node.getOrigin("reset"), nav_agent.getTermination("save_state"))
    net.connect(nav_term_node.getOrigin("reset"), nav_agent.getTermination("save_action"))


    ###CTRL AGENT
    actions = [("a", [0, 1]), ("b", [1, 0])] # actions corresponding to "go to A" or "go to B"
    ctrl_agent = smdpagent.SMDPAgent(stateN, len(env.placecells) + contextD, actions, name="CtrlAgent",
                                     state_encoders=enc, state_evals=evals, state_threshold=0.8,
                                     **ctrlargs)
    net.add(ctrl_agent)

    print "agent neurons:", ctrl_agent.countNeurons()

    # ctrl_agent gets environmental state and reward
    net.connect(env.getOrigin("placewcontext"), ctrl_agent.getTermination("state_input"))
    net.connect(env.getOrigin("reward"), ctrl_agent.getTermination("reward"))

    # termination node for ctrl_agent (terminates whenever the agent is in the state targeted by
    # the ctrl_agent)
    # also has a long timer so that ctrl_agent doesn't get permanently stuck in one action
    ctrl_term_node = terminationnode.TerminationNode({"a":[0, 1], "b":[1, 0], terminationnode.Timer((30, 30)):None},
                                                     env, contextD=2, name="CtrlTermNode", rewardval=1.5)
    net.add(ctrl_term_node)

    # reward for nav_agent is the pseudoreward from ctrl_agent termination
    net.connect(ctrl_term_node.getOrigin("pseudoreward"), nav_agent.getTermination("reward"))

    net.connect(ctrl_term_node.getOrigin("reset"), ctrl_agent.getTermination("reset"))
    net.connect(ctrl_term_node.getOrigin("learn"), ctrl_agent.getTermination("learn"))
    net.connect(ctrl_term_node.getOrigin("reset"), ctrl_agent.getTermination("save_state"))
    net.connect(ctrl_term_node.getOrigin("reset"), ctrl_agent.getTermination("save_action"))

    # connect ctrl_agent action to termination context
    # this is used so that ctrl_term_node knows what the current goal is (to determine termination
    # and pseudoreward)
    net.connect(ctrl_agent.getOrigin("action_output"), ctrl_term_node.getTermination("context"))

    # state input for nav_agent is the environmental state + the output of ctrl_agent
    ctrl_output_relay = net.make("ctrl_output_relay", 1, len(env.placecells) + contextD, mode="direct")
    ctrl_output_relay.fixMode()
    trans = list(MU.I(len(env.placecells))) + [[0 for _ in range(len(env.placecells))] for _ in range(contextD)]
    net.connect(env.getOrigin("place"), ctrl_output_relay, transform=trans)
    net.connect(ctrl_agent.getOrigin("action_output"), ctrl_output_relay,
                transform=[[0 for _ in range(contextD)] for _ in range(len(env.placecells))] + list(MU.I(contextD)))

    net.connect(ctrl_output_relay, nav_agent.getTermination("state_input"))

    # periodically save the weights
    weight_save = 600.0 # period to save weights (realtime, not simulation time)
    HRLutils.WeightSaveThread(nav_agent.getNode("QNetwork").saveParams,
                     os.path.join("weights", "%s_%s" % (nav_agent.name, tag)), weight_save).start()
    HRLutils.WeightSaveThread(ctrl_agent.getNode("QNetwork").saveParams,
                     os.path.join("weights", "%s_%s" % (ctrl_agent.name, tag)), weight_save).start()

    # data collection node
    data = datanode.DataNode(period=5, show_plots=None, filename=HRLutils.datafile("dataoutput_%s.txt" % tag))
    net.add(data)
    data.record(env.getOrigin("reward"), filter=1e-5)
    data.record(ctrl_agent.getNode("QNetwork").getNode("actionvals").getOrigin("X"), filter=1e-4, func=max)
    data.record(ctrl_agent.getNode("QNetwork").getNode("actionvals").getOrigin("X"), filter=1e-4, func=min)
    data.record_sparsity(ctrl_agent.getNode("QNetwork").getNode("state_pop").getOrigin("AXON"), filter=1e-4)
    data.record_avg(ctrl_agent.getNode("QNetwork").getNode("valdiff").getOrigin("X"), filter=1e-5)
    # ErrorNetwork is apparently not the correct name and hell if I know what the correct one is
    #data.record_avg(ctrl_agent.getNode("ErrorNetwork").getOrigin("error"), filter=1e-5)

    net.add_to_nengo()
#    net.view()
    net.run(10000)

def run_contextenvironment(args, seed=None):
    """Runs the model on the context task.

    :param args: kwargs for the agent
    :param seed: random seed
    """

    if seed is not None:
        HRLutils.set_seed(seed)
    seed = HRLutils.SEED

    net = nef.Network("runContextEnvironment")

    if args.has_key("load_weights") and args["load_weights"] is not None:
        args["load_weights"] += "_%s" % seed

    stateN = 1200 # number of neurons to use in state population
    contextD = 2 # dimension of context vector
    context_scale = 1.0 # scale of context representation vs state representation
    max_state_input = 2 # max length of input vector for state population
    actions = [("up", [0, 1]), ("right", [1, 0]), ("down", [0, -1]), ("left", [-1, 0])]
        # actions (label and vector) available to the system

    # context labels and rewards for achieving those context goals
    rewards = {"a":1.5, "b":1.5}

    env = contextenvironment.ContextEnvironment(actions, HRLutils.datafile("contextmap.bmp"),
                                                contextD, rewards,
                                                colormap={-16777216:"wall",
                                                           - 1:"floor",
                                                           - 256:"a",
                                                           - 2088896:"b"},
                                                imgsize=(5, 5), dx=0.001, placedev=0.5)
    net.add(env)

    print "generated", len(env.placecells), "placecells"

    # termination node for agent (just goes off on some regular interval)
    term_node = terminationnode.TerminationNode({terminationnode.Timer((0.6, 0.9)):0.0}, env)
    net.add(term_node)

    # generate encoders and divide by max_state_input (so that all inputs
    # will end up being radius 1)
    enc = env.gen_encoders(stateN, contextD, context_scale)
    enc = MU.prod(enc, 1.0 / max_state_input)

    # load eval points from file
    with open(HRLutils.datafile("contextbmp_evalpoints_%s.txt" % seed)) as f:
        print "loading contextbmp_evalpoints_%s.txt" % seed
        evals = [[float(x) for x in l.split(" ")] for l in f.readlines()]

    agent = smdpagent.SMDPAgent(stateN, len(env.placecells) + contextD, actions,
                                state_encoders=enc, state_evals=evals, state_threshold=0.8,
                                **args)
    net.add(agent)

    print "agent neurons:", agent.countNeurons()

    # periodically save the weights
    weight_save = 600.0 # period to save weights (realtime, not simulation time)
    HRLutils.WeightSaveThread(agent.getNode("QNetwork").saveParams,
                     os.path.join("weights", "%s_%s" % (agent.name, seed)), weight_save).start()


    # data collection node
    data = datanode.DataNode(period=5, show_plots=None, filename=HRLutils.datafile("dataoutput_%s.txt" % seed))
    net.add(data)
    data.record(env.getOrigin("reward"), filter=1e-5)
    data.record(agent.getNode("QNetwork").getNode("actionvals").getOrigin("X"), filter=1e-4, func=max)
    data.record(agent.getNode("QNetwork").getNode("actionvals").getOrigin("X"), filter=1e-4, func=min)
    data.record_sparsity(agent.getNode("QNetwork").getNode("state_pop").getOrigin("AXON"), filter=1e-4)
    data.record_avg(agent.getNode("QNetwork").getNode("valdiff").getOrigin("X"), filter=1e-5)
    data.record_avg(env.getOrigin("state"), filter=1e-4)

    net.connect(env.getOrigin("placewcontext"), agent.getTermination("state_input"))
    net.connect(env.getOrigin("reward"), agent.getTermination("reward"))
    net.connect(term_node.getOrigin("reset"), agent.getTermination("reset"))
    net.connect(term_node.getOrigin("learn"), agent.getTermination("learn"))
    net.connect(term_node.getOrigin("reset"), agent.getTermination("save_state"))
    net.connect(term_node.getOrigin("reset"), agent.getTermination("save_action"))

    net.connect(agent.getOrigin("action_output"), env.getTermination("action"))

    net.add_to_nengo()
#    net.view()
    net.run(2000)

def gen_evalpoints(filename, seed=None):
    """Runs an environment for some length of time and records state values,
    to be used as eval points for agent initialization.

    :param filename: name of file in which to save eval points
    :param seed: random seed
    """

    if seed is not None:
        HRLutils.set_seed(seed)
    seed = HRLutils.SEED

    net = nef.Network("gen_evalpoints")

    contextD = 2
    actions = [("up", [0, 1]), ("right", [1, 0]), ("down", [0, -1]), ("left", [-1, 0])]

    rewards = {"a":1, "b":1}

    env = contextenvironment.ContextEnvironment(actions, HRLutils.datafile("contextmap.bmp"),
                                                contextD, rewards,
                                                imgsize=(5, 5), dx=0.001, placedev=0.5,
                                                colormap={-16777216:"wall",
                                                           - 1:"floor",
                                                           - 256:"a",
                                                           - 2088896:"b"})

    net.add(env)

    stateD = len(env.placecells) + contextD
    actions = env.actions
    actionD = len(actions)

    class EvalRecorder(nef.SimpleNode):
        def __init__(self, evalfile):
            self.action = actions[0]
            self.evalpoints = []
            self.evalfile = evalfile

            nef.SimpleNode.__init__(self, "EvalRecorder")

        def tick(self):
            if self.t % 0.1 < 0.001:
                self.evalpoints += [self.state]

            if self.t % 10.0 < 0.001:
                if len(self.evalpoints) > 10000:
                    self.evalpoints = self.evalpoints[len(self.evalpoints) - 10000:]

                with open(self.evalfile, "w") as f:
                    f.write("\n".join([" ".join([str(x) for x in e]) for e in self.evalpoints]))

        def termination_state(self, x, dimensions=stateD):
            self.state = x

        def termination_action_in(self, x, dimensions=actionD):
            self.action = actions[x.index(max(x))]

        def origin_action_out(self):
            return self.action[1]

    em = EvalRecorder(HRLutils.datafile("%s_%s.txt" % (filename, seed)))
    net.add(em)

    net.connect(em.getOrigin("action_out"), env.getTermination("action"))
    net.connect(env.getOrigin("optimal_move"), em.getTermination("action_in"))
    net.connect(env.getOrigin("placewcontext"), em.getTermination("state"))

    net.add_to_nengo()
#    net.view()
    net.run(500)

def run_flat_delivery(args, seed=None):
    """Runs the model on the delivery task with only one hierarchical level."""

    if seed is not None:
        HRLutils.set_seed(seed)
    seed = HRLutils.SEED

    net = nef.Network("run_flat_delivery")

    if args.has_key("load_weights") and args["load_weights"] is not None:
        args["load_weights"] += "_%s" % seed

    stateN = 1200
    contextD = 2
    context_scale = 1.0
    max_state_input = 2
    actions = [("up", [0, 1]), ("right", [1, 0]), ("down", [0, -1]), ("left", [-1, 0])]

    ###ENVIRONMENT

    env = deliveryenvironment.DeliveryEnvironment(actions, HRLutils.datafile("contextmap.bmp"),
                                                  colormap={-16777216:"wall",
                                                           - 1:"floor",
                                                           - 256:"a",
                                                           - 2088896:"b"},
                                                  imgsize=(5, 5), dx=0.001, placedev=0.5)
    net.add(env)

    print "generated", len(env.placecells), "placecells"

    ###NAV AGENT

    enc = env.gen_encoders(stateN, contextD, context_scale)
    enc = MU.prod(enc, 1.0 / max_state_input)

    with open(HRLutils.datafile("contextbmp_evalpoints_%s.txt" % seed)) as f:
        evals = [[float(x) for x in l.split(" ")] for l in f.readlines()]

    nav_agent = smdpagent.SMDPAgent(stateN, len(env.placecells) + contextD, actions, name="NavAgent",
                                    state_encoders=enc, state_evals=evals, state_threshold=0.8,
                                    **args)
    net.add(nav_agent)

    print "agent neurons:", nav_agent.countNeurons()

    net.connect(nav_agent.getOrigin("action_output"), env.getTermination("action"))
    net.connect(env.getOrigin("placewcontext"), nav_agent.getTermination("state_input"))
#    net.connect(env.getOrigin("reward"), nav_agent.getTermination("reward"))
#    net.connect(env.getOrigin("optimal_move"), nav_agent.getTermination("bg_input"))

    nav_term_node = terminationnode.TerminationNode({terminationnode.Timer((0.6, 0.9)):None}, env,
                                                    name="NavTermNode", contextD=2)
    net.add(nav_term_node)
    net.connect(env.getOrigin("context"), nav_term_node.getTermination("context"))
    net.connect(nav_term_node.getOrigin("reset"), nav_agent.getTermination("reset"))
    net.connect(nav_term_node.getOrigin("learn"), nav_agent.getTermination("learn"))
    net.connect(nav_term_node.getOrigin("reset"), nav_agent.getTermination("save_state"))
    net.connect(nav_term_node.getOrigin("reset"), nav_agent.getTermination("save_action"))

    reward_relay = net.make("reward_relay", 1, 1, mode="direct")
    reward_relay.fixMode()
    net.connect(env.getOrigin("reward"), reward_relay)
    net.connect(nav_term_node.getOrigin("pseudoreward"), reward_relay)
    net.connect(reward_relay, nav_agent.getTermination("reward"))

    #save weights
    weight_save = 600.0 #period to save weights (realtime, not simulation time)
    HRLutils.WeightSaveThread(nav_agent.getNode("QNetwork").saveParams,
                     os.path.join("weights", "%s_%s" % (nav_agent.name, seed)), weight_save).start()

    #data collection node
    data = datanode.DataNode(period=5, show_plots=None, filename=HRLutils.datafile("dataoutput_%s.txt" % seed))
    net.add(data)
    data.record_avg(env.getOrigin("reward"), filter=1e-5)
    data.record_avg(nav_agent.getNode("QNetwork").getNode("actionvals").getOrigin("X"), filter=1e-5)
    data.record_sparsity(nav_agent.getNode("QNetwork").getNode("state_pop").getOrigin("AXON"), filter=1e-5)
    data.record_avg(nav_agent.getNode("QNetwork").getNode("valdiff").getOrigin("X"), filter=1e-5)
    # ErrorNetwork is apparently not the correct name and hell if I know what the correct one is
    #data.record_avg(nav_agent.getNode("ErrorNetwork").getOrigin("error"), filter=1e-5)

    net.add_to_nengo()
    net.view()
#    net.run(10000)

def run_badreenvironment(nav_args, ctrl_args, seed=None, flat=False):
    
    if seed is not None:
        HRLutils.set_seed(seed)
    seed = HRLutils.SEED
    
    net = nef.Network("run_badreenvironment")

    env = badreenvironment.BadreEnvironment(flat=flat)
    net.add(env)

    ###NAV AGENT
    stateN = 500
    max_state_input = 2
    enc = env.gen_encoders(stateN, 0, 1.0)
    enc = MU.prod(enc, 1.0 / max_state_input)

#    with open(HRLutils.datafile("badre_evalpoints.txt")) as f:
#        evals = [[float(x) for x in l.split(" ")] for l in f.readlines()]
    orientations = MU.I(env.num_orientations)
    shapes = MU.I(env.num_shapes)
    colours = MU.I(env.num_colours)
    evals = list(MU.I(env.stateD)) + \
            [o+s+c for o in orientations for s in shapes for c in colours]

    nav_agent = smdpagent.SMDPAgent(stateN, env.stateD,
                                    env.actions, name="NavAgent",
                                    load_weights=None,
                                    state_encoders=enc, state_evals=evals,
                                    discount=0.4, **nav_args)
    net.add(nav_agent)

    print "agent neurons:", nav_agent.countNeurons()

    nav_term_node = terminationnode.TerminationNode({terminationnode.Timer((0.6, 0.6)):None}, env,
                                                    name="NavTermNode", state_delay=0.1)
    net.add(nav_term_node)

    net.connect(nav_term_node.getOrigin("reset"), nav_agent.getTermination("reset"))
    net.connect(nav_term_node.getOrigin("learn"), nav_agent.getTermination("learn"))
    net.connect(nav_term_node.getOrigin("reset"), nav_agent.getTermination("save_state"))
    net.connect(nav_term_node.getOrigin("reset"), nav_agent.getTermination("save_action"))

    net.connect(nav_agent.getOrigin("action_output"), env.getTermination("action"))

    ###CTRL AGENT
    enc = env.gen_encoders(stateN, 0, 0)
    enc = MU.prod(enc, 1.0 / max_state_input)
    actions = [("shape", [0, 1]), ("orientation", [1, 0]), ("null", [0, 0])]
    ctrl_agent = smdpagent.SMDPAgent(stateN, env.stateD, actions, name="CtrlAgent",
                                     load_weights=None, state_encoders=enc,
                                     state_evals=evals, discount=0.4, **ctrl_args)
    net.add(ctrl_agent)

    print "agent neurons:", ctrl_agent.countNeurons()

    net.connect(env.getOrigin("state"), ctrl_agent.getTermination("state_input"))

    ctrl_term_node = terminationnode.TerminationNode({terminationnode.Timer((0.6, 0.6)):None},
                                                     env, name="CtrlTermNode",
                                                     state_delay=0.1)
    net.add(ctrl_term_node)

    net.connect(ctrl_term_node.getOrigin("reset"), ctrl_agent.getTermination("reset"))
    net.connect(ctrl_term_node.getOrigin("learn"), ctrl_agent.getTermination("learn"))
    net.connect(ctrl_term_node.getOrigin("reset"), ctrl_agent.getTermination("save_state"))
    net.connect(ctrl_term_node.getOrigin("reset"), ctrl_agent.getTermination("save_action"))
    
    
    ## reward for nav/ctrl
    reward_relay = net.make("reward_relay", 1, 2, mode="direct")
    reward_relay.fixMode()
    net.connect(env.getOrigin("reward"), reward_relay, transform=[[1], [0]])
    net.connect(ctrl_agent.getOrigin("action_output"), reward_relay, transform=[[0, 0], [1, 1]])
    
    # nav reward is just environment
    net.connect(reward_relay, nav_agent.getTermination("reward"), 
                func=lambda x: x[0], origin_name="nav_reward")
    
    # ctrl gets a slight bonus if it selects a rule (as opposed to null), to encourage it not
    # to just pick null all the time
    net.connect(reward_relay, ctrl_agent.getTermination("reward"), 
                func=lambda x: x[0]+0.25*abs(x[0]) if x[1] > 0.5 else x[0], origin_name="ctrl_reward")

    ## state for navagent controlled by ctrlagent
#    ctrl_output_relay = net.make("ctrl_output_relay", 1, env.stateD+2, mode="direct")
#    ctrl_output_relay.fixMode()
    ctrl_output_relay = net.make_array("ctrl_output_relay", 50, env.stateD,
                                       radius=2, mode=HRLutils.SIMULATION_MODE)
    ctrl_output_relay.fixMode([SimulationMode.DEFAULT, SimulationMode.RATE])
    
    inhib_matrix = [[0,-5]]*50*env.num_orientations + \
                   [[-5,0]]*50*env.num_shapes + \
                   [[-5,-5]]*50*env.num_colours

    # ctrl output inhibits all the non-selected aspects of the state
    net.connect(env.getOrigin("state"), ctrl_output_relay)
    net.connect(ctrl_agent.getOrigin("action_output"), ctrl_output_relay,
#                transform=zip([0]*env.num_orientations + [-1]*(env.num_shapes+env.num_colours),
#                              [-1]*env.num_orientations + [0]*env.num_shapes + [-1]*env.num_colours))
                transform=inhib_matrix)
    
    # also give a boost to the selected aspects (so that neurons are roughly equally activated).
    # adding 2/3 to each element (base vector has length 3, inhibited vector has length 1, so add 2/3*3 --> 3)
    net.connect(ctrl_agent.getOrigin("action_output"), ctrl_output_relay,
                transform=zip([0.66]*env.num_orientations + [0]*(env.num_shapes+env.num_colours),
                              [0]*env.num_orientations + [0.66]*env.num_shapes + [2]*env.num_colours))

    net.connect(ctrl_output_relay, nav_agent.getTermination("state_input"))

    # save weights
    weight_save = 600.0 # period to save weights (realtime, not simulation time)
    HRLutils.WeightSaveThread(nav_agent.getNode("QNetwork").saveParams,
                     os.path.join("weights", "%s_%s" % (nav_agent.name, seed)), weight_save).start()
    HRLutils.WeightSaveThread(ctrl_agent.getNode("QNetwork").saveParams,
                     os.path.join("weights", "%s_%s" % (ctrl_agent.name, seed)), weight_save).start()

    # data collection node
    data = datanode.DataNode(period=5, show_plots=None, filename=HRLutils.datafile("dataoutput_%s.txt" % seed))
    filter = 1e-5
    net.add(data)
    data.record_avg(env.getOrigin("reward"), filter=filter)
    data.record_avg(ctrl_agent.getNode("QNetwork").getNode("actionvals").getOrigin("X"), filter=filter)
    data.record_sparsity(ctrl_agent.getNode("QNetwork").getNode("state_pop").getOrigin("AXON"), filter=filter)
    data.record_sparsity(nav_agent.getNode("QNetwork").getNode("state_pop").getOrigin("AXON"), filter=filter)
    data.record_avg(ctrl_agent.getNode("QNetwork").getNode("valdiff").getOrigin("X"), filter=filter)
    data.record_avg(ctrl_agent.getNode("ErrorNetwork").getOrigin("error"), filter=filter)
    data.record_avg(ctrl_agent.getNode("BGNetwork").getNode("weight_actions").getNode("0").getOrigin("AXON"), filter=filter)
    data.record_avg(ctrl_agent.getNode("BGNetwork").getNode("weight_actions").getNode("1").getOrigin("AXON"), filter=filter)

    net.add_to_nengo()
#    net.view()
    net.run(2000)

NodeThreadPool.setNumJavaThreads(8) #Set it equal to the number of cores?

#run_deliveryenvironment({"learningrate": 9e-10, "discount": 0.1, "Qradius": 2.0,
#                         "load_weights": os.path.join("weights", "contextgrid_decoder", "SMDPAgent")},
#                        {"learningrate": 9e-10, "discount": 0.1, "load_weights": None},
#                        seed=1)

#run_contextenvironment({"learningrate":9e-10, "discount":0.1, "Qradius":2.0,
#                        "load_weights":None},
#                       seed=0)

#gen_evalpoints("contextbmp_evalpoints", seed=0)

#run_flat_delivery({"learningrate":9e-10, "discount":0.1, "Qradius":2.0,
#                   "load_weights":os.path.join("delivery", "flat", "NavAgent")},
#                  seed=2)

run_flat_delivery({"learningrate":9e-10, "manual_control":True, "discount":0.1, "Qradius":2.0,
                   "load_weights":os.path.join("delivery", "flat", "NavAgent")},
                  seed=2)


#run_badreenvironment({"learningrate":9e-7, "state_threshold":0.8}, #9e-10
#                     {"learningrate":7e-7, "state_threshold":0.0}, #7e-10
#                     seed=0, flat=False)