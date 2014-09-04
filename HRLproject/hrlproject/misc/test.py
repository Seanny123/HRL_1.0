from __future__ import with_statement

"""Miscellaneous test functions for prototyping various bits of
functionality."""

import sys
import socket
import math
import os

compname = socket.gethostname()
if compname == "CTN04" or compname == "DANIEL-PC":
    sys.path.append(r"D:\Documents\officesvn\HRLproject")
elif compname == "ctngpu2":
    sys.path.append("/home/ctnuser/drasmuss/HRLproject")
elif compname == "hybrid":
    sys.path.append("/home/sean/HRL_link")
    sys.path.append("/home/sean/HRL_link/HRLproject/hrlproject/misc")
    sys.path.append("/home/sean/HRL_link/HRLproject/hrlproject")
    sys.path.append("/home/sean/HRL_link/HRLproject")
elif compname == "ctn11":
    sys.path.append("/home/saubin/Github/HRL_1.0")
    sys.path.append("/home/saubin/Github/HRL_1.0/HRLproject/hrlproject")
    sys.path.append("/home/saubin/Github/HRL_1.0/HRLproject")
else:  # assume running on sharcnet
    compname = "sharcnet"
    sys.path.append("/home/drasmuss/HRLproject")

#from misc import HRLutils
#HRLutils.full_reset()

from hrlproject.misc import HRLutils, gridworldwatch, boxworldwatch
from hrlproject.agent import smdpagent, errorcalc, actionvalues
from hrlproject.environment import (gridworldenvironment, boxenvironment,
                                    contextenvironment, deliveryenvironment,
                                    placecell_bmp, badreenvironment,
                                    pongenvironment)
from hrlproject.simplenodes import (datanode, terminationnode, errornode,
                                    badre_pseudoreward, decoderlearningnode)
from hrlproject.misc.HRLutils import rand as random

from ca.nengo.model import Units, SimulationMode
from ca.nengo.model.impl import (NetworkImpl, FunctionInput)
from ca.nengo.math.impl import (ConstantFunction, PiecewiseConstantFunction,
                                IdentityFunction, IndicatorPDF)
from ca.nengo.util import MU #Matrix utlities

import nef
import timeview
from nef.templates import hpes_termination

# This method doesn't appear to be used anymore
def test_errorcalc():
    test = NetworkImpl()
    test.name = "testErrorCalc"

    gamma = 0.95

    e = errorcalc.ErrorCalc(gamma)
    test.addNode(e)

    #test with Q change
#    reset = FunctionInput("reset", [PiecewiseConstantFunction([0.1], [1.0, 0.0])], Units.UNK)
#    Q = FunctionInput("Q", [PiecewiseConstantFunction([0.5], [0.5, 1.0])], Units.UNK)
#    reward = FunctionInput("reward", [ConstantFunction(1,0)], Units.UNK)

    #test with reward change
#    reset = FunctionInput("reset", [PiecewiseConstantFunction([0.1], [1.0, 0.0])], Units.UNK)
#    Q = FunctionInput("Q", [PiecewiseConstantFunction([0.5], [0.5, 0.5])], Units.UNK)
#    reward = FunctionInput("reward", [PiecewiseConstantFunction([0.5,0.7],[0.0,1.0,0.0])], Units.UNK)

    #test with equal Q/reward change
    reset = FunctionInput("reset", [PiecewiseConstantFunction([0.1], [1.0, 0.0])], Units.UNK) #UNK means unknown
    Q = FunctionInput("Q", [PiecewiseConstantFunction([0.7], [0.5, 0.3])], Units.UNK)
    reward = FunctionInput("reward", [PiecewiseConstantFunction([0.3, 0.5], [0.0, 1.0, 0.0])], Units.UNK)

    #test with combined Q/reward change
#    reset = FunctionInput("reset", [PiecewiseConstantFunction([0.1], [1.0, 0.0])], Units.UNK)
#    Q = FunctionInput("Q", [PiecewiseConstantFunction([0.7], [0.5, 0.7])], Units.UNK)
#    reward = FunctionInput("reward", [PiecewiseConstantFunction([0.3,0.5],[0.0,1.0,0.0])], Units.UNK)


    test.addNode(reset)
    test.addNode(Q)
    test.addNode(reward)
    test.addProjection(reset.getOrigin("origin"), e.getTermination("reset"))
    test.addProjection(Q.getOrigin("origin"), e.getTermination("currQ"))
    test.addProjection(reward.getOrigin("origin"), e.getTermination("reward"))

#    world.add(test) # Don't know what "world" is here

def test_decoderlearning():
    net = nef.Network("test_decoderlearning")


    learningrate = 1e-8
    N = 100

    fin1 = net.make_fourier_input('fin1', base=0.1, high=10, power=0.5, seed=12)
    fin2 = net.make_fourier_input('fin2', base=0.1, high=10, power=0.5, seed=13)

    pre = net.make("pre", N, 2)
    net.connect(fin1, pre, transform=[[1], [0]])
    net.connect(fin2, pre, transform=[[0], [1]])

    err = net.make("err", N, 2)
    net.connect(fin1, err, transform=[[1], [0]])
    net.connect(fin2, err, transform=[[0], [1]])
    net.connect(pre, err, func=lambda x: [0.0, 0.0], transform=[[-1, 0], [0, -1]])

    dlnode = decoderlearningnode.DecoderLearningNode(pre, pre.getOrigin("<lambda>"), learningrate, errorD=2)
    net.add(dlnode)

    net.connect(err, dlnode.getTermination("error"))

    learn = net.make_input("learn", [1])
    net.connect(learn, dlnode.getTermination("learn"))

    net.network.setMode(SimulationMode.RATE)

    net.add_to_nengo()
    net.view()

def test_learning():
    test = NetworkImpl()
    test.name = "testLearning"
    learningrate = 5e-4
    N = 500

    ef = HRLutils.defaultEnsembleFactory()

    pre = ef.make("pre", N, 1)
    pre.addDecodedTermination("input", [[1]], 0.01, False)
    test.addNode(pre)

    post = ef.make("post", N, 1)
    test.addNode(post)

    hpes_termination.make(nef.Network(test), errName="err1", preName="pre", postName="post", rate=learningrate, supervisionRatio=1.0)

    #flattened decoder origin
    decoders = [x[0] for x in post.getOrigin("X").getDecoders()]
    encoders = [x[0] for x in post.getEncoders()]
    posdec = [decoders[i] for i, e in enumerate(encoders) if e >= 0]
    negdec = [decoders[i] for i, e in enumerate(encoders) if e < 0]
    posmean = sum(posdec) / len(posdec)
    negmean = sum(negdec) / len(negdec)
    new_decoders = [[posmean] if e >= 0 else [negmean] for e in encoders]
    o = post.addDecodedOrigin("flatX", [IdentityFunction()], "AXON")
    o.setDecoders(new_decoders)



    nef.Network(test).make_fourier_input('fin', base=0.1, high=10, power=0.5, seed=12)
    fin = test.getNode("fin")
    test.addProjection(fin.getOrigin("origin"), pre.getTermination("input"))


    err = test.getNode("err1")
    err.addDecodedTermination("input", [[-1]], 0.01, False)
    err.addDecodedTermination("target", [[1]], 0.01, False)
#    test.addProjection(post.getOrigin("X"), err.getTermination("input"))
    test.addProjection(post.getOrigin("flatX"), err.getTermination("input"))
    test.addProjection(fin.getOrigin("origin"), err.getTermination("target"))

#    test.setMode(SimulationMode.RATE)

#    world.add(test)

def test_actionvalues():
    net = nef.Network("testActionValues")

    stateN = 200
    N = 100
    stateD = 2
    stateradius = 1.0
    statelength = math.sqrt(2 * stateradius ** 2)
    init_Qs = 0.5
    learningrate = 0.0
    Qradius = 1
    tauPSC = 0.007
    actions = [("up", [0, 1]), ("right", [1, 0]), ("down", [0, -1]), ("left", [-1, 0])]

    #state
    state_pop = net.make("state_pop", stateN, stateD,
                              radius=statelength,
                              node_factory=HRLutils.node_fac(),
                              eval_points=[[x / statelength, y / statelength] for x in range(-int(stateradius), int(stateradius))
                                             for y in range(-int(stateradius), int(stateradius))])
    state_pop.fixMode([SimulationMode.DEFAULT, SimulationMode.RATE])
    state_pop.addDecodedTermination("state_input", MU.I(stateD), tauPSC, False)

    #set up action nodes
    decoders = state_pop.addDecodedOrigin("init_decoders", [ConstantFunction(stateD, init_Qs)], "AXON").getDecoders()

    actionvals = actionvalues.ActionValues(N, stateN, actions, learningrate, Qradius=Qradius, init_decoders=decoders)
    net.add(actionvals)

    net.connect(state_pop.getOrigin("AXON"), actionvals.getTermination("curr_state"))

    #input
    inp = net.make_input("input", [0, 0])
    net.connect(inp, state_pop.getTermination("state_input"))

    net.add_to_nengo()

# gets weird "colour grid error"
def test_gridworld():
    net = nef.Network("testGridWorld")

    stateN = 400
    stateD = 2
    actions = [("up", [0, 1]), ("right", [1, 0]), ("down", [0, -1]), ("left", [-1, 0])]

    agent = smdpagent.SMDPAgent(stateN, stateD, actions, Qradius=1, stateradius=3, rewardradius=1, learningrate=9e-10)
    net.add(agent)

#    agent.loadWeights("weights\\potjansgrid")

    env = gridworldenvironment.GridWorldEnvironment(stateD, actions, HRLutils.datafile("smallgrid.txt"),
                                                    cartesian=True, delay=(0.6, 0.9), datacollection=False)
    net.add(env)

    net.connect(env.getOrigin("state"), agent.getTermination("state_input"))
    net.connect(env.getOrigin("reward"), agent.getTermination("reward"))
    net.connect(env.getOrigin("reset"), agent.getTermination("reset"))
    net.connect(env.getOrigin("learn"), agent.getTermination("learn"))
    net.connect(env.getOrigin("reset"), agent.getTermination("save_state"))
    net.connect(env.getOrigin("reset"), agent.getTermination("save_action"))

    net.connect(agent.getOrigin("action_output"), env.getTermination("action"))
    net.connect(agent.getOrigin("Qs"), env.getTermination("Qs"))

    net.add_to_nengo()
    view = timeview.View(net.network, update_frequency=5)
    view.add_watch(gridworldwatch.GridWorldWatch())
    view.restore()

def test_boxworld():
    net = nef.Network("testBoxWorld")

    stateN = 400
    stateD = 2
    actions = [("up", [0, 1]), ("right", [1, 0]), ("down", [0, -1]), ("left", [-1, 0])]

    env = boxenvironment.BoxEnvironment(stateD, actions, HRLutils.datafile("smallgrid.txt"), cartesian=True,
                                         delay=(0.6, 0.9), cellwidth=1.0, dx=0.0015)
    net.add(env)

    agent = smdpagent.SMDPAgent(stateN, stateD, actions, Qradius=1, stateradius=3, rewardradius=1, learning=True)
    net.add(agent)

#    agent.loadWeights("weights\\smallgrid")

    net.connect(env.getOrigin("state"), agent.getTermination("state_input"))
    net.connect(env.getOrigin("reward"), agent.getTermination("reward"))
    net.connect(env.getOrigin("reset"), agent.getTermination("reset"))
    net.connect(env.getOrigin("learn"), agent.getTermination("learn"))
    net.connect(env.getOrigin("reset"), agent.getTermination("save_state"))
    net.connect(env.getOrigin("reset"), agent.getTermination("save_action"))

    net.connect(agent.getOrigin("action_output"), env.getTermination("action"))
#    net.connect(agent.getOrigin("Qs"), env.getTermination("Qs"))

    net.add_to_nengo()
    view = timeview.View(net.network, update_frequency=5)
    view.add_watch(boxworldwatch.BoxWorldWatch())
    view.restore()

def test_placecellenvironment():
    net = nef.Network("testBoxWorld")

    stateN = 800
#    stateD = 2
    actions = [("up", [0, 1]), ("right", [1, 0]), ("down", [0, -1]), ("left", [-1, 0])]

    env = placecell_bmp.PlaceCellEnvironment(actions, HRLutils.datafile("smallgrid.txt"),
                                         delay=(0.6, 0.9), cellwidth=1.0, dx=0.001, placedev=0.5)
    net.add(env)

    print "generated", len(env.placecells), "placecells"

    agent = smdpagent.SMDPAgent(stateN, len(env.placecells), actions, learningrate=3e-9, #4e-9, 
                                manual_control=False, load_weights=None,
#                                state_encoders=MU.I(len(env.placecells)))
                                state_encoders=env.gen_encoders(stateN))
    net.add(agent)

    #data collection node
    data = datanode.DataNode(period=5, show_plots=None, filename=HRLutils.datafile("dataoutput.txt"))
    filter = 1e-4
    net.add(data)
    data.record_avg(env.getOrigin("reward"), filter=filter)
    data.record_avg(agent.getNode("QNetwork").getNode("actionvals").getOrigin("X"), filter=filter)
    data.record_sparsity(agent.getNode("QNetwork").getNode("state_pop").getOrigin("AXON"), filter=filter)

#    net.connect(env.getOrigin("state"), agent.getTermination("state_input"))
    net.connect(env.getOrigin("place"), agent.getTermination("state_input"))
    net.connect(env.getOrigin("reward"), agent.getTermination("reward"))
    net.connect(env.getOrigin("reset"), agent.getTermination("reset"))
    net.connect(env.getOrigin("learn"), agent.getTermination("learn"))
    net.connect(env.getOrigin("reset"), agent.getTermination("save_state"))
    net.connect(env.getOrigin("reset"), agent.getTermination("save_action"))

    net.connect(agent.getOrigin("action_output"), env.getTermination("action"))
#    net.connect(agent.getOrigin("Qs"), env.getTermination("Qs"))

    net.add_to_nengo()
    view = timeview.View(net.network, update_frequency=5)
    view.add_watch(boxworldwatch.BoxWorldWatch())
    view.restore()

def test_sparsestate():
    net = nef.Network("test_sparsestate")

    state_fac = HRLutils.node_fac()
    state_fac.setIntercept(IndicatorPDF(0, 1))

    actions = [("up", [0, 1]), ("right", [1, 0]), ("down", [0, -1]), ("left", [-1, 0])]
    env = placecell_bmp.PlaceCellEnvironment(actions, HRLutils.datafile("smallgrid.txt"),
                                         delay=(0.6, 0.9), cellwidth=0.1, dx=0.001, placedev=0.5, num_places=10)

    N = 500
    d = 10
    state_pop = net.make("state_pop", N, d,
                          radius=1,
                          node_factory=state_fac,
                          encoders=env.gen_encoders(N))
#                          encoders = MU.I(d))
    state_pop.addDecodedOrigin("init_decoders", [ConstantFunction(d, 0.5)], "AXON")

    input = net.make_input("input", [0 for _ in range(d)])

    net.connect(input, state_pop)

    net.add_to_nengo()
    net.view()

def test_contextenvironment():
#    NodeThreadPool.setNumJavaThreads(1)

    net = nef.Network("testContextEnvironment")

    stateN = 1200
#    stateD = 2
    contextD = 2
    context_scale = 1.0
    max_state_input = 2
    actions = [("up", [0, 1]), ("right", [1, 0]), ("down", [0, -1]), ("left", [-1, 0])]

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

    term_node = terminationnode.TerminationNode({terminationnode.Timer((0.6, 0.9)):0.0}, env)
    net.add(term_node)

    enc = env.gen_encoders(stateN, contextD, context_scale)
    enc = MU.prod(enc, 1.0 / max_state_input)
    with open(HRLutils.datafile("contextgrid_evalpoints.txt")) as f:
        evals = [[float(x) for x in l.split(" ")] for l in f.readlines()]

    agent = smdpagent.SMDPAgent(stateN, len(env.placecells) + contextD, actions,
                                manual_control=False, optimal_control=False,
                                learningrate=9e-10, load_weights=None,
                                stateradius=1.0, state_encoders=enc, state_evals=evals,
                                discount=0.1, rewardradius=1, Qradius=2, state_threshold=0.8)
    net.add(agent)

    print "agent neurons:", agent.countNeurons()

    #periodically save the weights
    weight_save = 600.0 #period to save weights (realtime, not simulation time)
    HRLutils.WeightSaveThread(agent.getNode("QNetwork").saveParams,
                     os.path.join("weights", "%s" % (agent.name)), weight_save).start()


    #data collection node
    data = datanode.DataNode(period=5, show_plots=None, filename=HRLutils.datafile("dataoutput.txt"))
    net.add(data)
    data.record(env.getOrigin("reward"), filter=1e-5)
    data.record(agent.getNode("QNetwork").getNode("actionvals").getOrigin("X"), filter=1e-4, func=max)
    data.record(agent.getNode("QNetwork").getNode("actionvals").getOrigin("X"), filter=1e-4, func=min)
    data.record_sparsity(agent.getNode("QNetwork").getNode("state_pop").getOrigin("AXON"), filter=1e-4)
    data.record_avg(agent.getNode("QNetwork").getNode("valdiff").getOrigin("X"), filter=1e-5)
    data.record_avg(env.getOrigin("state"), filter=1e-4)

#    net.connect(env.getOrigin("state"), agent.getTermination("state_input"))
    net.connect(env.getOrigin("placewcontext"), agent.getTermination("state_input"))
    net.connect(env.getOrigin("reward"), agent.getTermination("reward"))
    net.connect(term_node.getOrigin("reset"), agent.getTermination("reset"))
    net.connect(term_node.getOrigin("learn"), agent.getTermination("learn"))
    net.connect(term_node.getOrigin("reset"), agent.getTermination("save_state"))
    net.connect(term_node.getOrigin("reset"), agent.getTermination("save_action"))
#    net.connect(env.getOrigin("optimal_move"), agent.getTermination("bg_input"))

    net.connect(agent.getOrigin("action_output"), env.getTermination("action"))
#    net.connect(agent.getOrigin("Qs"), env.getTermination("Qs"))

    net.add_to_nengo()
    net.view()
#    view = timeview.View(net.network, update_frequency=5)
#    view.add_watch(boxworldwatch.BoxWorldWatch())
#    view.restore()

def test_intercepts():
    net = nef.Network("test_intercepts")

#    contextD = 2
#    context_scale=1.0
#    actions = [("up",[0,1]), ("right",[1,0]), ("down",[0,-1]), ("left",[-1,0])]
#    
#    rewards = {"a":1, "b":1}

#    env = contextenvironment.ContextEnvironment(actions, HRLutils.datafile("contextmap.bmp"),
#                                                contextD, rewards,
#                                                imgsize=(5,5), dx=0.001, placedev=0.5,
#                                                colormap={-16777216:"wall",
#                                                           -1:"floor",
#                                                           -256:"a",
#                                                           -2088896:"b"})
#    env = deliveryenvironment.DeliveryEnvironment(actions, HRLutils.datafile("contextmap.bmp"),
#                                         imgsize=(5,5), dx=0.001, placedev=0.5,
#                                         colormap={-16777216:"wall",
#                                                   -1:"floor",
#                                                   -256:"a",
#                                                   -2088896:"b"})

    env = badreenvironment.BadreEnvironment(flat=True)

    net.add(env)

#    stateD = len(env.placecells)+contextD
#    actionD = 4
#    encoders = env.gen_encoders(1000, contextD, context_scale)

    stateD = env.stateD
    actionD = 3
    encoders = env.gen_encoders(1000, 0, 0)
    actions = env.actions

    class EncoderMult(nef.SimpleNode):
        def __init__(self, encoders, evalfile):
            self.enc = encoders
            self.min = 1000
            self.max = -1000
            self.action = actions[0]
            self.evalpoints = []
            self.evalfile = evalfile

            nef.SimpleNode.__init__(self, "EncoderMult")

        def tick(self):
            prod = MU.prod(self.enc, self.state)
            self.min = min(self.min, min(prod))
            self.max = max(self.max, max(prod))
#            print self.min, self.max

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

        def origin_results(self):
            return [self.min, self.max]

    em = EncoderMult(encoders, HRLutils.datafile("badre_evalpoints.txt"))
    net.add(em)

    net.connect(em.getOrigin("action_out"), env.getTermination("action"))
    net.connect(env.getOrigin("optimal_move"), em.getTermination("action_in"))
#    net.connect(env.getOrigin("placewcontext"), em.getTermination("state"))
    net.connect(env.getOrigin("state"), em.getTermination("state"))

    net.add_to_nengo()
    net.view()

def test_deliveryenvironment():
    net = nef.Network("testDeliveryEnvironment")

    stateN = 1200 # The number of neurons in the population representing the state
    contextD = 2
    context_scale = 1.0
    max_state_input = 2
    actions = [("up", [0, 1]), ("right", [1, 0]), ("down", [0, -1]), ("left", [-1, 0])]

    ###ENVIRONMENT

    # Instantiate the delivery environment class, which holds the place-cells and the reward, I think?
    env = deliveryenvironment.DeliveryEnvironment(actions, HRLutils.datafile("contextmap.bmp"),
                                                  colormap={-16777216:"wall",
                                                           - 1:"floor",
                                                           - 256:"a",
                                                           - 2088896:"b"},
                                                  imgsize=(5, 5), dx=0.001, placedev=0.5)
    # Add the generated environment to the network
    net.add(env)

    print "generated", len(env.placecells), "placecells"



    ###NAV AGENT

    enc = env.gen_encoders(stateN, contextD, context_scale)
    enc = MU.prod(enc, 1.0 / max_state_input) #Wtf is MU?

    with open(HRLutils.datafile("contextgrid_evalpoints.txt")) as f:
        evals = [[float(x) for x in l.split(" ")] for l in f.readlines()]

    nav_agent = smdpagent.SMDPAgent(stateN, len(env.placecells) + contextD, actions, name="NavAgent",
                                manual_control=False, optimal_control=False,
                                learningrate=9e-10, load_weights=os.path.join("weights", "NavAgent"),
                                stateradius=1.0, state_encoders=enc, state_evals=evals,
                                discount=0.1, rewardradius=1, Qradius=2.0, state_threshold=0.8)
    net.add(nav_agent)

    print "agent neurons:", nav_agent.countNeurons()

    nav_term_node = terminationnode.TerminationNode({terminationnode.Timer((0.6, 0.9)):None}, env,
                                                    contextD=2, name="NavTermNode")
    net.add(nav_term_node)

    net.connect(nav_term_node.getOrigin("reset"), nav_agent.getTermination("reset"))
    net.connect(nav_term_node.getOrigin("learn"), nav_agent.getTermination("learn"))
    net.connect(nav_term_node.getOrigin("reset"), nav_agent.getTermination("save_state"))
    net.connect(nav_term_node.getOrigin("reset"), nav_agent.getTermination("save_action"))
#    net.connect(env.getOrigin("optimal_move"), agent.getTermination("bg_input"))

    net.connect(nav_agent.getOrigin("action_output"), env.getTermination("action"))

    ###CTRL AGENT
    actions = [("a", [0, 1]), ("b", [1, 0])]
    ctrl_agent = smdpagent.SMDPAgent(stateN, len(env.placecells) + contextD, actions, name="CtrlAgent",
                                     manual_control=False, optimal_control=False,
                                     learningrate=9e-10, load_weights=os.path.join("weights", "CtrlAgent"),
                                     stateradius=1.0, state_encoders=enc, state_evals=evals,
                                     discount=0.1, rewardradius=1, state_threshold=0.8)
    net.add(ctrl_agent)

    print "agent neurons:", ctrl_agent.countNeurons()

    ctrl_term_node = terminationnode.TerminationNode({"a":[0, 1], "b":[1, 0], terminationnode.Timer((30, 30)):None},
                                                     env, contextD=2, name="CtrlTermNode", rewardval=1.5)
    net.add(ctrl_term_node)
    net.connect(ctrl_term_node.getOrigin("pseudoreward"), nav_agent.getTermination("reward"))

    net.connect(env.getOrigin("placewcontext"), ctrl_agent.getTermination("state_input"))
    net.connect(env.getOrigin("reward"), ctrl_agent.getTermination("reward"))
    net.connect(ctrl_term_node.getOrigin("reset"), ctrl_agent.getTermination("reset"))
    net.connect(ctrl_term_node.getOrigin("learn"), ctrl_agent.getTermination("learn"))
    net.connect(ctrl_term_node.getOrigin("reset"), ctrl_agent.getTermination("save_state"))
    net.connect(ctrl_term_node.getOrigin("reset"), ctrl_agent.getTermination("save_action"))
#    net.connect(env.getOrigin("optimal_move"), agent.getTermination("bg_input"))

    net.connect(ctrl_agent.getOrigin("action_output"), ctrl_term_node.getTermination("context"))
        #this is used so that ctrl_term_node knows what the current goal is
#    net.connect(ctrl_agent.getOrigin("action_output"), nav_term_node.getTermination("context"))
#        #this is just used so that nav_term_node knows to reset when the context changes

    ctrl_output_relay = net.make("ctrl_output_relay", 1, len(env.placecells) + contextD, mode="direct")
    ctrl_output_relay.fixMode()
    trans = list(MU.I(len(env.placecells))) + [[0 for _ in range(len(env.placecells))] for _ in range(contextD)]
    net.connect(env.getOrigin("place"), ctrl_output_relay, transform=trans)
    net.connect(ctrl_agent.getOrigin("action_output"), ctrl_output_relay,
                transform=[[0 for _ in range(contextD)] for _ in range(len(env.placecells))] + list(MU.I(contextD)))

    net.connect(ctrl_output_relay, nav_agent.getTermination("state_input"))


    #data collection node
    data = datanode.DataNode(period=5, show_plots=None, filename=HRLutils.datafile("dataoutput.txt"))
    filter = 1e-5
    net.add(data)
    data.record(env.getOrigin("reward"), filter=1e-5)
    data.record(ctrl_agent.getNode("QNetwork").getNode("actionvals").getOrigin("X"), filter=1e-4, func=max)
    data.record(ctrl_agent.getNode("QNetwork").getNode("actionvals").getOrigin("X"), filter=1e-4, func=min)
    data.record_sparsity(ctrl_agent.getNode("QNetwork").getNode("state_pop").getOrigin("AXON"), filter=1e-4)
    data.record_avg(ctrl_agent.getNode("QNetwork").getNode("valdiff").getOrigin("X"), filter=1e-5)
#    data.record_avg(ctrl_agent.getNode("ErrorNetwork").getOrigin("error"), filter=filter)

    net.add_to_nengo()
    net.view()

def test_terminationnode():
    net = nef.Network("testTerminationNode")

    actions = [("up", [0, 1]), ("right", [1, 0]), ("down", [0, -1]), ("left", [-1, 0])]
    env = deliveryenvironment.DeliveryEnvironment(actions, HRLutils.datafile("contextmap.bmp"),
                                                  colormap={-16777216:"wall",
                                                           - 1:"floor",
                                                           - 256:"a",
                                                           - 2088896:"b"},
                                                  imgsize=(5, 5), dx=0.001, placedev=0.5)
    net.add(env)

    term_node = terminationnode.TerminationNode({"a":[0, 1], "b":[1, 0], terminationnode.Timer((30, 30)):None},
                                                env, contextD=2, rewardval=1)
    net.add(term_node)

    print term_node.conds

    context_input = net.make_input("contextinput", {0.0:[0, 0.1], 0.5:[1, 0], 1.0:[0, 1]})
    net.connect(context_input, term_node.getTermination("context"))

    net.add_to_nengo()
    net.view()

def test_bmp():
    from javax.imageio import ImageIO
    from java.io import File
    from java.awt import Rectangle

    img = ImageIO.read(File(HRLutils.datafile("contextmap.bmp")))

    colours = [int(val) for val in img.getRGB(0, 0, img.getWidth(), img.getHeight(), None, 0, img.getWidth())]
    unique_colours = []
    for c in colours:
        if not c in unique_colours:
            unique_colours += [c]

    print unique_colours

#    g2d = img.createGraphics()
##    g2d.setPaint(
#    g2d.drawRect(0,0,10,10)
#    ImageIO.write(img, "bmp", File(HRLutils.datafile("test_output.bmp")))

def test_placecell_bmp():
    net = nef.Network("TestPlacecellBmp")

    stateN = 800
#    stateD = 2
    actions = [("up", [0, 1]), ("right", [1, 0]), ("down", [0, -1]), ("left", [-1, 0])]

    env = placecell_bmp.PlaceCellEnvironment(actions, HRLutils.datafile("contextmap.bmp"),
                                             colormap={-16777216:"wall",
                                                       - 1:"floor",
                                                       - 256:"target",
                                                       - 2088896:"b"},
                                             imgsize=(5, 5), dx=0.001, placedev=0.5)
    net.add(env)

    print "generated", len(env.placecells), "placecells"

#    agent = smdpagent.SMDPAgent(stateN, len(env.placecells), actions, learningrate=4e-10,
#                                manual_control=False, load_weights=None,
#                                state_encoders=env.gen_encoders(stateN))
#    net.add(agent)
#
##    net.connect(env.getOrigin("state"), agent.getTermination("state_input"))
#    net.connect(env.getOrigin("place"), agent.getTermination("state_input"))
#    net.connect(env.getOrigin("reward"), agent.getTermination("reward"))
##    net.connect(env.getOrigin("reset"), agent.getTermination("reset"))
##    net.connect(env.getOrigin("learn"), agent.getTermination("learn"))
##    net.connect(env.getOrigin("reset"), agent.getTermination("save_state"))
##    net.connect(env.getOrigin("reset"), agent.getTermination("save_action"))
#    
#    net.connect(agent.getOrigin("action_output"), env.getTermination("action"))
##    net.connect(agent.getOrigin("Qs"), env.getTermination("Qs"))

    net.add_to_nengo()
    net.view()

def test_errornode():
    net = nef.Network("test_errornode")

    error_net = errornode.ErrorNode(4, discount=0.3)
    net.add(error_net)

    net.make_input("reset", [0])
    net.make_input("learn", [0])
    net.make_input("reward", [0])

    net.connect("reset", error_net.getTermination("reset"))
    net.connect("learn", error_net.getTermination("learn"))
    net.connect("reward", error_net.getTermination("reward"))

    net.add_to_nengo()
    net.view()

def test_flat_delivery():
    net = nef.Network("testFlatDeliveryEnvironment")

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
    net.add(env) # Billy: this is in a with block now

    print "generated", len(env.placecells), "placecells"

    ###NAV AGENT

    enc = env.gen_encoders(stateN, contextD, context_scale)
    enc = MU.prod(enc, 1.0 / max_state_input)

#    print [len(e) for e in enc]

    with open(HRLutils.datafile("contextgrid_evalpoints.txt")) as f:
        evals = [[float(x) for x in l.split(" ")] for l in f.readlines()]

    nav_agent = smdpagent.SMDPAgent(stateN, len(env.placecells) + contextD, actions, name="NavAgent",
                                manual_control=False, optimal_control=False,
                                learningrate=9e-10, load_weights=None,
                                stateradius=1.0, state_encoders=enc, state_evals=evals,
                                discount=0.1, rewardradius=1, Qradius=2.0, state_threshold=0.8)
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
                     os.path.join("weights", "%s" % (nav_agent.name)), weight_save).start()

    #data collection node
    data = datanode.DataNode(period=5, show_plots=None, filename=HRLutils.datafile("dataoutput.txt"))
    filter = 1e-5
    net.add(data)
    data.record_avg(env.getOrigin("reward"), filter=filter)
    data.record_avg(nav_agent.getNode("QNetwork").getNode("actionvals").getOrigin("X"), filter=filter)
    data.record_sparsity(nav_agent.getNode("QNetwork").getNode("state_pop").getOrigin("AXON"), filter=filter)
    data.record_avg(nav_agent.getNode("QNetwork").getNode("valdiff").getOrigin("X"), filter=filter)
    data.record_avg(nav_agent.getNode("ErrorNetwork").getOrigin("error"), filter=filter)



    net.add_to_nengo()
    net.view()


def test_badreenvironment():
    net = nef.Network("test_badreenvironment")

    env = badreenvironment.BadreEnvironment(flat=False)
    net.add(env)

    ###NAV AGENT
    stateN = 500
#    contextD = 2
    max_state_input = 2
    enc = env.gen_encoders(stateN, 0, 1.0)
    enc = MU.prod(enc, 1.0 / max_state_input)

    with open(HRLutils.datafile("badre_evalpoints.txt")) as f:
        evals = [[float(x) for x in l.split(" ")] for l in f.readlines()]

#    contexts = list(MU.I(contextD)) + [[0.3 for _ in range(contextD)]]

    nav_agent = smdpagent.SMDPAgent(stateN, env.stateD,
                                    env.actions, name="NavAgent",
                                    learningrate=9e-10, load_weights=None,
                                    state_encoders=enc,
#                                    state_evals=[x + list(random.choice(contexts))
#                                                 for x in evals],
                                    state_evals = evals,
                                    discount=0.3, state_threshold=0.8)
    net.add(nav_agent)

    print "agent neurons:", nav_agent.countNeurons()

    nav_term_node = terminationnode.TerminationNode({terminationnode.Timer((0.6, 0.6)):None}, env,
                                                    name="NavTermNode", state_delay=0.1)
    net.add(nav_term_node)

    net.connect(nav_term_node.getOrigin("reset"), nav_agent.getTermination("reset"))
    net.connect(nav_term_node.getOrigin("learn"), nav_agent.getTermination("learn"))
    net.connect(nav_term_node.getOrigin("reset"), nav_agent.getTermination("save_state"))
    net.connect(nav_term_node.getOrigin("reset"), nav_agent.getTermination("save_action"))
#    net.connect(env.getOrigin("optimal_move"), agent.getTermination("bg_input"))

    net.connect(nav_agent.getOrigin("action_output"), env.getTermination("action"))

    ###CTRL AGENT
    enc = env.gen_encoders(stateN, 0, 0)
    enc = MU.prod(enc, 1.0 / max_state_input)
    actions = [("shape", [0, 1]), ("orientation", [1, 0]), ("null", [0, 0])]
    ctrl_agent = smdpagent.SMDPAgent(stateN, env.stateD, actions, name="CtrlAgent",
                                     learningrate=7e-10, load_weights=None, state_encoders=enc,
                                     state_evals=evals, discount=0.1)
    net.add(ctrl_agent)

    print "agent neurons:", ctrl_agent.countNeurons()

    net.connect(env.getOrigin("state"), ctrl_agent.getTermination("state_input"))
#    net.connect(env.getOrigin("reward"), ctrl_agent.getTermination("reward"))

    ctrl_term_node = terminationnode.TerminationNode({terminationnode.Timer((0.6, 0.6)):None},
                                                     env, name="CtrlTermNode",
                                                     state_delay=0.1)
    net.add(ctrl_term_node)

    net.connect(ctrl_term_node.getOrigin("reset"), ctrl_agent.getTermination("reset"))
    net.connect(ctrl_term_node.getOrigin("learn"), ctrl_agent.getTermination("learn"))
    net.connect(ctrl_term_node.getOrigin("reset"), ctrl_agent.getTermination("save_state"))
    net.connect(ctrl_term_node.getOrigin("reset"), ctrl_agent.getTermination("save_action"))
#    net.connect(env.getOrigin("optimal_move"), agent.getTermination("bg_input"))

#    rewardnode = badre_pseudoreward.BadrePseudoreward(env, actions)
#    net.add(rewardnode)
#    net.connect(ctrl_agent.getOrigin("action_output"), rewardnode.getTermination("context"))
#    net.connect(nav_agent.getOrigin("action_output"), rewardnode.getTermination("action"))

#    reward_relay = net.make("reward_relay", 1, 2, mode="direct") 
#    reward_relay.fixMode()
#    net.connect(rewardnode.getOrigin("pseudoreward"), reward_relay, transform=[[0.66], [0]])
#    net.connect(env.getOrigin("reward"), reward_relay, transform=[[0.33], [0]])
#    net.connect(rewardnode.getOrigin("pseudoreward"), reward_relay, transform=[[0], [0.33]])
#    net.connect(env.getOrigin("reward"), reward_relay, transform=[[0], [0.66]])
#        #note: the first dimension represents a combined reward with greater emphasis on 
#        #pseudoreward; this will be used as input to the low level. the second dimension
#        #is the opposite, and will be used as input to the top level.
#        #the important part is the relative weighting. just multiplying everything by 2 to
#        #get larger reward magnitudes, and more differentiated Q values.

#    reward_relay = net.make("reward_relay", 1, 2, mode="direct")
#    reward_relay.fixMode()
#    net.connect(rewardnode.getOrigin("pseudoreward"), reward_relay, transform=[[1], [0], [0]])
#    net.connect(env.getOrigin("reward"), reward_relay, transform=[[0], [1]])

    # reward for navagent: 2/3 pseudoreward + 1/3 envreward (thresholded at +/- 2)
    # we want the agent to be primarily driven by the ctrlagent commands (i.e. pseudoreward)
    # even when pseudoreward disagrees with envreward. that is why pseudoreward needs
    # a greater weight than envreward
#    net.connect(reward_relay, nav_agent.getTermination("reward"), origin_name="out0",
#                func=lambda x: x[0] if abs(x[0]) > 0.01 else x[1])
    
    # reward for ctrlagent: if there is pseudoreward (i.e. the ctrlagent has selected a 
    # goal), ctrlagent is rewarded when pseudoreward matches envreward.  that is,
    # ctrlagent is rewarded when its commands are consistent with env results. if the
    # ctrlagent has selected no goal, just use envreward
#    net.connect(reward_relay, ctrl_agent.getTermination("reward"), origin_name="out1",
#                func=lambda x: x[0] * x[1] * 0.5 if abs(x[0]) > 0.01 else x[1] * 0.5)
    
    
    reward_relay = net.make("reward_relay", 1, 2, mode="direct")
    reward_relay.fixMode()
    net.connect(env.getOrigin("reward"), reward_relay, transform=[[1], [0]])
    net.connect(ctrl_agent.getOrigin("action_output"), reward_relay, transform=[[0, 0], [1, 1]])
    
    net.connect(reward_relay, nav_agent.getTermination("reward"), 
                func=lambda x: x[0], origin_name="nav_reward")
    net.connect(reward_relay, ctrl_agent.getTermination("reward"), 
                func=lambda x: x[0]+0.33*abs(x[0]) if x[1] > 0.5 else x[0], origin_name="ctrl_reward")



    ctrl_output_relay = net.make("ctrl_output_relay", 1, env.stateD, mode="direct")
    ctrl_output_relay.fixMode()
#    net.connect(env.getOrigin("state"), ctrl_output_relay,
#                transform=list(MU.I(env.stateD)) + [[0 for _ in range(env.stateD)] for _ in range(contextD)])
#    net.connect(ctrl_agent.getOrigin("action_output"), ctrl_output_relay,
#                transform=[[0 for _ in range(contextD)] for _ in range(env.stateD)] + list(MU.I(contextD)))
    
    net.connect(env.getOrigin("state"), ctrl_output_relay)
    tmp = zip([0]*env.num_orientations + [-1.5]*(env.num_shapes+env.num_colours),
              [-1.5]*env.num_orientations + [0]*env.num_shapes + [-1.5]*env.num_colours)
    net.connect(ctrl_agent.getOrigin("action_output"), ctrl_output_relay,
                transform=tmp)

    net.connect(ctrl_output_relay, nav_agent.getTermination("state_input"))

    #save weights
    weight_save = 600.0 #period to save weights (realtime, not simulation time)
    HRLutils.WeightSaveThread(nav_agent.getNode("QNetwork").saveParams,
                     os.path.join("weights", "%s" % (nav_agent.name)), weight_save).start()
    HRLutils.WeightSaveThread(ctrl_agent.getNode("QNetwork").saveParams,
                     os.path.join("weights", "%s" % (ctrl_agent.name)), weight_save).start()

    #data collection node
    data = datanode.DataNode(period=5, show_plots=None, filename=HRLutils.datafile("dataoutput.txt"))
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
    net.view()

def test_pongenvironment():
    net = nef.Network("test_pongenvironment")
    
    actions = [("up", [1, 0, 0]), ("stay", [0, 1, 0]), ("down", [0, 0, 1])]
    
    env = pongenvironment.PongEnvironment(actions)
    net.add(env)
    
    stateN = 800
    nav_agent = smdpagent.SMDPAgent(stateN, len(env.placecells), actions, name="PongAgent",
                                learningrate=3e-8, load_weights=None,
                                stateradius=2.0,
                                state_encoders=env.gen_encoders(stateN),
                                state_evals=[env.calc_activations(env.random_location(),
                                                                   env.place_dev)
                                             for _ in range(len(env.placecells) * 10)],
                                discount=0.1, rewardradius=1, Qradius=1.0, state_threshold=0.0)
    net.add(nav_agent)
    
    net.connect(env.getOrigin("place"), nav_agent.getTermination("state_input"))
    net.connect(env.getOrigin("reward"), nav_agent.getTermination("reward"))
    net.connect(nav_agent.getOrigin("action_output"), env.getTermination("action"))
    
    nav_term_node = terminationnode.TerminationNode({terminationnode.Timer((0.35, 0.35)):None}, env,
                                                    name="NavTermNode", state_delay=0.0, learn_interval=0.05,
                                                    reset_delay=0.05, reset_interval=0.05)
    net.add(nav_term_node)

    net.connect(nav_term_node.getOrigin("reset"), nav_agent.getTermination("reset"))
    net.connect(nav_term_node.getOrigin("learn"), nav_agent.getTermination("learn"))
    net.connect(nav_term_node.getOrigin("reset"), nav_agent.getTermination("save_state"))
    net.connect(nav_term_node.getOrigin("reset"), nav_agent.getTermination("save_action"))
    
    #save weights
    weight_save = 600.0 #period to save weights (realtime, not simulation time)
    HRLutils.WeightSaveThread(nav_agent.getNode("QNetwork").saveParams,
                     os.path.join("weights", "%s" % (nav_agent.name)), weight_save).start()
    
    #data collection node
    data = datanode.DataNode(period=5, show_plots=None, filename=HRLutils.datafile("dataoutput.txt"))
    filter = 1e-5
    net.add(data)
    data.record(env.getOrigin("reward"), filter=filter)
    data.record_sparsity(nav_agent.getNode("QNetwork").getNode("state_pop").getOrigin("AXON"), filter=filter)
    data.record(env.getOrigin("stats"))
    data.record(nav_agent.getNode("QNetwork").getOrigin("vals"), filter=filter)
    data.record(nav_agent.getNode("QNetwork").getOrigin("old_vals"), filter=filter)
    
    env.start_pong()
    
    net.add_to_nengo()
    net.view(play=1000)
    
def test_pongenvironment_hier():
    net = nef.Network("test_pongenvironment")
    
    ctrl_actions = [("top", [0.8]),
                    ("midtop", [0.4]),
                    ("mid", [0.0]),
                    ("midbot", [-0.4]),
                    ("bot", [-0.8])]
    nav_actions = [("up", [0, 0, 0]), ("stay", [0, 1, 0]), ("down", [0, 0, 1])]
    
    env = pongenvironment.PongEnvironment(nav_actions)
    net.add(env)
    
    stateN = 800
    ctrl_agent = smdpagent.SMDPAgent(stateN, 2, ctrl_actions, name="PongCtrl",
                                learningrate=3e-8, load_weights=None,
                                stateradius=1.0,
#                                state_encoders=env.gen_encoders(stateN),
#                                state_evals=[env.calc_activations(env.random_location(),
#                                                                   env.place_dev)
#                                             for _ in range(len(env.placecells) * 10)],
                                discount=0.1, rewardradius=1, Qradius=1.0, state_threshold=0.0)
    net.add(ctrl_agent)
    
    ctrlstate = net.make("ctrlstate", 1, 1, mode="direct")
    ctrlstate.fixMode()
    
    net.connect(env.getOrigin("state"), ctrlstate, transform=[[1, 0]])
    net.connect(ctrlstate, ctrl_agent.getTermination("state_input"), function=lambda x: [math.sin(x*math.pi), math.cos(x*math.pi)])
    net.connect(env.getOrigin("reward"), ctrl_agent.getTermination("reward"))
    
    ctrl_term_node = terminationnode.TerminationNode({terminationnode.Timer((0.35, 0.35)):None}, env,
                                                    name="CtrlTermNode", state_delay=0.0, learn_interval=0.05,
                                                    reset_delay=0.05, reset_interval=0.05)
    net.add(ctrl_term_node)

    net.connect(ctrl_term_node.getOrigin("reset"), ctrl_agent.getTermination("reset"))
    net.connect(ctrl_term_node.getOrigin("learn"), ctrl_agent.getTermination("learn"))
    net.connect(ctrl_term_node.getOrigin("reset"), ctrl_agent.getTermination("save_state"))
    net.connect(ctrl_term_node.getOrigin("reset"), ctrl_agent.getTermination("save_action"))
    
    nav_agent = smdpagent.SMDPAgent(stateN, 3, nav_actions, name="PongNav",
                                learningrate=3e-8, load_weights=None,
                                stateradius=1.0,
#                                state_encoders=env.gen_encoders(stateN),
#                                state_evals=[env.calc_activations(env.random_location(),
#                                                                   env.place_dev)
#                                             for _ in range(len(env.placecells) * 10)],
                                discount=0.1, rewardradius=1, Qradius=1.0, state_threshold=0.0)
    net.add(nav_agent)
    
    navstate = net.make("navstate", 1, 2, mode="direct")
    navstate.fixMode()
    
    net.connect(env.getOrigin("state"), navstate, transform=[[0, 1], [0, 0]])
    net.connect(ctrl_agent.getOrigin("action_output"), navstate, transform=[[0], [1]])
    net.connect(navstate, nav_agent.getTermination("state_input"), function=lambda x: [math.sin(x[0])*math.cos(x[1]),
                                                                                       math.sin(x[0])*math.sin(x[1]),
                                                                                       math.cos(x[0])])

    net.connect(navstate, nav_agent.getTermination("reward"), function=lambda x: 0.5 if abs(x[0] - x[1]) < 0.2 else -0.05)
    
    net.connect(nav_agent.getOrigin("action_output"), env.getTermination("action"))
    
    nav_term_node = terminationnode.TerminationNode({terminationnode.Timer((0.35, 0.35)):None}, env,
                                                    name="NavTermNode", state_delay=0.0, learn_interval=0.05,
                                                    reset_delay=0.05, reset_interval=0.05)
    net.add(nav_term_node)

    net.connect(nav_term_node.getOrigin("reset"), nav_agent.getTermination("reset"))
    net.connect(nav_term_node.getOrigin("learn"), nav_agent.getTermination("learn"))
    net.connect(nav_term_node.getOrigin("reset"), nav_agent.getTermination("save_state"))
    net.connect(nav_term_node.getOrigin("reset"), nav_agent.getTermination("save_action"))
    
    #save weights
    weight_save = 600.0 #period to save weights (realtime, not simulation time)
    HRLutils.WeightSaveThread(nav_agent.getNode("QNetwork").saveParams,
                     os.path.join("weights", "%s" % (nav_agent.name)), weight_save).start()
    HRLutils.WeightSaveThread(ctrl_agent.getNode("QNetwork").saveParams,
                     os.path.join("weights", "%s" % (ctrl_agent.name)), weight_save).start()
    
    #data collection node
    data = datanode.DataNode(period=5, show_plots=None, filename=HRLutils.datafile("dataoutput.txt"))
    filter = 1e-5
    net.add(data)
    data.record(env.getOrigin("reward"), filter=filter)
    data.record_sparsity(nav_agent.getNode("QNetwork").getNode("state_pop").getOrigin("AXON"), filter=filter)
    data.record_sparsity(ctrl_agent.getNode("QNetwork").getNode("state_pop").getOrigin("AXON"), filter=filter)
    data.record(env.getOrigin("stats"))
    data.record(nav_agent.getNode("QNetwork").getOrigin("vals"), filter=filter)
    data.record(ctrl_agent.getNode("QNetwork").getOrigin("vals"), filter=filter)
    
    env.start_pong()
    
    net.add_to_nengo()
    net.view(play=1000)
    
test_gridworld()
#test_decoderlearning()
#test_placecellenvironment()
#test_sparsestate()
#test_contextenvironment()
#test_intercepts()
#test_deliveryenvironment()
#test_terminationnode()
#test_bmp()
#test_placecell_bmp()
#test_errornode()
#test_flat_delivery()
#test_badreenvironment()
#test_decoderlearning()
#test_pongenvironment()
#test_pongenvironment_hier()

#from misc import pdb
#pdb.run("test_contextenvironment()")
