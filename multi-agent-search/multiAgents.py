# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
 
        #Calculating distance all ghosts and pacman and if checking if there is immediate danger
        ghostsDist = 1
        danger = 0
        for ghost in successorGameState.getGhostPositions():
            distance = util.manhattanDistance(newPos, ghost)
            ghostsDist += distance
            if distance <= 1:
                danger += 1

       #Calculating distance to the farthest food pellet
        FoodList = newFood.asList() #list of all pellets
        closestPelletDist = 10000
        for pellet in FoodList:
            distance = util.manhattanDistance(newPos, pellet)
            if closestPelletDist >= distance or closestPelletDist == 100000:
                closestPelletDist = distance

        #Current score + Closest food distance reciprocal - Distance to ghosts reciprocal - danger(priority)
        eval = successorGameState.getScore() + (1 / float(closestPelletDist)) - (1 / float(ghostsDist)) - danger
        return eval
    
def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        Here are some method calls that might be useful when implementing minimax.
            gameState.getLegalActions(agentIndex):
                Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1
            gameState.generateSuccessor(agentIndex, action):
                Returns the successor game state after an agent takes an action
            gameState.getNumAgents():
                Returns the total number of agents in the game
            gameState.isWin():
                Returns whether or not the game state is a winning state
            gameState.isLose():
                Returns whether or not the game state is a losing state"""
        "*** YOUR CODE HERE ***"
        #returns score, action pair
        action = self.minimax(gameState, 0, 0)
        return action[1]

    def minimax(self, gameState, index, depth):
        #Evaluates min/max agents and the terminal state
        # win/loss state:
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return gameState.getScore(), 

        # checks for pacman (max)
        if index == 0:
            return self.maxVal(gameState, index, depth)

        # Min(ghost) always has index above 0
        else:
            return self.minVal(gameState, index, depth)

    def maxVal(self, gameState, index, depth):
        #return best score/action pair
        legalMoves = gameState.getLegalActions(index)
        maxVal = float("-inf") #Python interprets this is negative infinity
        maxAct = "" #initalize best action variable

        for action in legalMoves: #evaluates viability of successive actions
            zucc = gameState.generateSuccessor(index, action)
            zuccIdx = index + 1
            zuccDepth = depth

            #Checks for pacman and updates idx and depth
            if zuccIdx == gameState.getNumAgents():
                zuccIdx = 0
                zuccDepth += 1

            currVal = self.minimax(zucc, zuccIdx, zuccDepth)[0]

            if currVal > maxVal:
                maxVal = currVal
                maxAct = action

        return maxVal, maxAct

    def minVal(self, gameState, index, depth):
        #returns best score/action pair
        legalMoves = gameState.getLegalActions(index)
        minVal = float("inf") #Python interprets as positive infinity
        minAct = "" #initialize best action variable

        for action in legalMoves:#evaluates viability of successive actions
            zucc = gameState.generateSuccessor(index, action)
            zuccIdx = index + 1
            zuccDepth = depth

            #Checks for pacman and updates idx and depth
            if zuccIdx == gameState.getNumAgents():
                zuccIdx = 0
                zuccDepth += 1

            currVal = self.minimax(zucc, zuccIdx, zuccDepth)[0]

            if currVal < minVal:
                minVal = currVal
                minAct = action

        return minVal, minAct

    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        "*** YOUR CODE HERE ***"
        #returns score, action pair
        # Returns best action with alpha as -infinity and beta as infinity
        result = self.alphabeta(gameState, 0, 0, float("-inf"), float("inf"))

        # Return the action from result
        return result[0]

    def alphabeta(self, gameState, index, depth, alpha, beta):
        #Evaluates min/max agents and the terminal state
        # win/loss state:
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return "", gameState.getScore()

        # checks for pacman (max)
        if index == 0:
            return self.maxVal(gameState, index, depth, alpha, beta)

        # if not pacman then ghost
        else:
            return self.minVal(gameState, index, depth, alpha, beta)

    def maxVal(self, gameState, index, depth, alpha, beta):
            #return best score/action pair with AB pruning
        legalMoves = gameState.getLegalActions(index)
        maxVal = float("-inf")
        maxAct = ""

        for action in legalMoves:
            zucc = gameState.generateSuccessor(index, action)
            zuccIdx = index + 1
            zuccDepth = depth

            #Checks for pacman and updates idx and depth
            if zuccIdx == gameState.getNumAgents():
                zuccIdx = 0
                zuccDepth += 1

            # Calculate the action-score for the current successor
            currAct, currVal \
                = self.alphabeta(zucc, zuccIdx, zuccDepth, alpha, beta)

            # Update maxVal and maxAct for maximizer agent
            if currVal > maxVal:
                maxVal = currVal
                maxAct = action

            # Update alpha value for current maximizer
            alpha = max(alpha, maxVal)


            if maxVal > beta:
                return maxAct, maxVal

        return maxAct, maxVal

    def minVal(self, gameState, index, depth, alpha, beta):
        #return best score/action pair with AB pruning
        legalMoves = gameState.getLegalActions(index)
        minVal = float("inf")
        minAct = ""

        for action in legalMoves:
            zucc = gameState.generateSuccessor(index, action)
            zuccIdx = index + 1
            zuccDepth = depth

            #Checks for pacman and updates idx and depth
            if zuccIdx == gameState.getNumAgents():
                zuccIdx = 0
                zuccDepth += 1

            # Calculate the action-score for the current successor
            currAct, currVal \
                = self.alphabeta(zucc, zuccIdx, zuccDepth, alpha, beta)

            # Update minVal and minAct for min agent
            if currVal < minVal:
                minVal = currVal
                minAct = action

            # Update beta value for current min agent
            beta = min(beta, minVal)

            if minVal < alpha:
                return minAct, minVal

        return minAct, minVal
        
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Format of result = [action, score]
        action, score = self.getVal(gameState, 0, 0)

        return action

    def getVal(self, gameState, index, depth):
        #Evaluates max agent expecti agent and the terminal state

        # win/loss state:
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return "", self.evaluationFunction(gameState)

        # checks for pacman (max)
        if index == 0:
            return self.maxVal(gameState, index, depth)

        # ExpectatiAgent
        else:
            return self.expVal(gameState, index, depth)

    def maxVal(self, gameState, index, depth):

        #return best score/action pair for max agent

        legalMoves = gameState.getLegalActions(index)
        maxVal = float("-inf")
        maxAct = ""

        for action in legalMoves:  
            zucc = gameState.generateSuccessor(index, action)
            zuccIdx = index + 1
            zuccDepth = depth

            #Checks for pacman and updates idx and depth
            if zuccIdx == gameState.getNumAgents():
                zuccIdx = 0
                zuccDepth += 1

            currAct, currVal = self.getVal(zucc, zuccIdx, zuccDepth)

            if currVal > maxVal:
                maxVal = currVal
                maxAct = action

        return maxAct, maxVal

    def expVal(self, gameState, index, depth):
        #return best score/action pair for max agent
        legalMoves = gameState.getLegalActions(index)
        expVal = 0
        expAct = ""

        # Find the current successor's probability using a uniform distribution
        zuccProb = 1.0 / len(legalMoves)

        for action in legalMoves:
            zucc = gameState.generateSuccessor(index, action)
            zuccIdx = index + 1
            zuccDepth = depth

            #Checks for pacman and updates idx and depth
            if zuccIdx == gameState.getNumAgents():
                zuccIdx = 0
                zuccDepth += 1

            # Calculate the action-score for the current successor
            currAct, currVal = self.getVal(zucc, zuccIdx, zuccDepth)

            # Update expVal with the currVal and zuccProb
            expVal += zuccProb * currVal

        return expAct, expVal

