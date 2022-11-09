# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    """
    "*** YOUR CODE HERE ***"
    # creating a Stack fringe to store nodes
    fringe = util.Stack()
    # list that tracks visited nodes
    visited = []
    # pushing start state to the fringe
    fringe.push((problem.getStartState(), 0, [])) #node, cost, path
    while not fringe.isEmpty():
        node,cost,path = fringe.pop()
        # goal check
        if problem.isGoalState(node):
            return path
        if node not in visited:
            visited.append(node)
            # visit child nodes
            for nextNode, nextAction, nextCost in problem.getSuccessors(node):
                # updates state
                totalCost = cost + nextCost
                totalPath = path + [nextAction]
                newState = (nextNode, totalCost, totalPath)
                fringe.push(newState)
    #fails if the fringe is empty and goal state is not found
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # creating a Queue fringe to store nodes
    fringe = util.Queue()
    # list that tracks visited nodes
    visited = []
    # pushing start state to the fringe
    fringe.push((problem.getStartState(), 0, []))
    while not fringe.isEmpty():
        node,cost,path = fringe.pop()
        # goal check
        if problem.isGoalState(node):
            return path
        if node not in visited:
            visited.append(node)
            # visit child nodes
            for nextNode, nextAction, nextCost in problem.getSuccessors(node):
                # updates state
                totalCost = cost + nextCost
                totalPath = path + [nextAction]
                newState = (nextNode, totalCost, totalPath)
                fringe.push(newState)
    #fails if the fringe is empty and goal state is not found
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first. """
    "*** YOUR CODE HERE ***"
    # Creating a PriorityQueue fringe to implement Uniform Cost,
    fringe = util.PriorityQueue()
    # list that tracks visited nodes
    visited = []
    # Pushing starting point to fringe
    fringe.push((problem.getStartState(), 0, []), 0)
    while not fringe.isEmpty():
        node, cost, path = fringe.pop()
        if problem.isGoalState(node):
                return path
        if node not in visited:
            visited.append(node)
            #visit child nodes
            for nextNode, nextAction, nextCost in problem.getSuccessors(node):
                # updates state
                totalCost = cost + nextCost
                totalPath = path + [nextAction]
                newState = (nextNode, totalCost, totalPath)
                fringe.push(newState, totalCost)
    #fails if the fringe is empty and goal state is not found
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # creating a Stack fringe to store nodes
    fringe = util.PriorityQueue()
    # list that tracks visited nodes
    visited = []
    # pushing start state to the fringe
    initialState = (problem.getStartState(), 0, [])#node, cost, path
    initialCost = 0 + heuristic(initialState[0], problem)
    fringe.push(initialState, initialCost) 
    while not fringe.isEmpty():
        node,cost,path = fringe.pop()
        # goal check
        if problem.isGoalState(node):
            return path
        if node not in visited:
            visited.append(node)
            # visit child nodes
            for nextNode, nextAction, nextCost in problem.getSuccessors(node):
                # updates state
                totalCost = cost + nextCost
                totalPath = path + [nextAction]
                newState = (nextNode, totalCost, totalPath)
                totalCost = totalCost + heuristic(newState[0], problem)
                fringe.push(newState, totalCost)
    #fails if the fringe is empty and goal state is not found
    util.raiseNotDefined()
    
    
    
    """
    start = problem.getStartState()
    exploredState = []
    states = util.PriorityQueue()
    states.push((start, []), nullHeuristic(start, problem))
    nCost = 0
    while not states.isEmpty():
        state, actions = states.pop()
        if problem.isGoalState(state):
            return actions
        if state not in exploredState:
            successors = problem.getSuccessors(state)
            for succ in successors:
                coordinates = succ[0]
                if coordinates not in exploredState:
                    directions = succ[1]
                    nActions = actions + [directions]
                    nCost = problem.getCostOfActions(nActions) + heuristic(coordinates, problem)
                    states.push((coordinates, actions + [directions]), nCost)
        exploredState.append(state)
    return actions
"""
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
