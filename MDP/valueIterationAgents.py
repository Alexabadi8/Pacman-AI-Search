# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            Q_values = self.values.copy()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                action = self.computeActionFromValues(state)
                Q = self.computeQValueFromValues(state, action)
                Q_values[state] = Q
            self.values = Q_values


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        trans = self.mdp.getTransitionStatesAndProbs(state, action)
        Q = 0
        for nextState, probs in trans:
            nextReward = self.mdp.getReward(state, action, nextState)
            discount = self.discount
            futureQVal = self.getValue(nextState)
            Q += probs * (nextReward + discount * futureQVal)
        return Q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        possibleActions = self.mdp.getPossibleActions(state)
        actionQValues = util.Counter()
        for possibleAction in possibleActions:
            actionQValues[possibleAction] = self.computeQValueFromValues(state, possibleAction)

        action = actionQValues.argMax()
        return action
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        iter = 0
        for i in range(self.iterations):
            if iter == len(states): 
                iter = 0
            newState = states[iter]
            iter += 1
            if self.mdp.isTerminal(newState):
                continue
            action = self.computeActionFromValues(newState)
            Q = self.computeQValueFromValues(newState, action)
            self.values[newState] = Q
        

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        self.queue = util.PriorityQueue()
        self.predecessors = util.Counter()
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                self.predecessors[state] = set()

        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue

            # implementing priority queue priority values
            currentValue = self.values[state]
            action = self.computeActionFromValues(state)
            highestQValue = self.computeQValueFromValues(state, action)
            diff = abs(currentValue - highestQValue)
            self.queue.push(state, -diff)
            # computing predecessors for current state
            possibleActions = self.mdp.getPossibleActions(state)
            for act in possibleActions:
                trans = self.mdp.getTransitionStatesAndProbs(state, act)
                for nextState, prob in trans:
                    if prob != 0 and not self.mdp.isTerminal(nextState):
                        self.predecessors[nextState].add(state)

        for i in range(self.iterations):
            if self.queue.isEmpty():
                # terminate
                return
 
            state = self.queue.pop()

            # calculate Q-value to update state
            action = self.computeActionFromValues(state)
            self.values[state] = self.computeQValueFromValues(state, action)

            for pred in self.predecessors[state]:
                currentValue = self.values[pred]
                action = self.computeActionFromValues(pred)
                highestQValue = self.computeQValueFromValues(pred, action)
                diff = abs(currentValue - highestQValue)
                if diff > self.theta:
                    self.queue.update(pred, -diff)
        

