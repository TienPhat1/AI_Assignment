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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        allFood = newFood.asList()
        newScore = successorGameState.getScore()

        minDistance = -1
        for food in allFood:
            if minDistance > util.manhattanDistance(newPos,food) or minDistance == -1:
                minDistance = util.manhattanDistance(newPos,food)

        totalghostDistance = 1
        mayDie = 0
        for ghost in successorGameState.getGhostPositions():
            ghostDistance = util.manhattanDistance(newPos,ghost)
            totalghostDistance += ghostDistance
            if ghostDistance <=1 :
                mayDie +=1

        return newScore + (1 / float(minDistance)) - (1 / float(totalghostDistance)) - mayDie


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
        """
        "*** YOUR CODE HERE ***"
        def minimax(agent,depth,state):
            if state.isWin() or state.isLose() or depth==self.depth:
                return self.evaluationFunction(state)
            if agent== 0:
                return max(minimax(1,depth,state.generateSuccessor(0,_state)) for _state in state.getLegalActions(0))
            else:
                newAgent = agent + 1
                if newAgent == state.getNumAgents():
                    newAgent=0
                if newAgent==0:
                    depth+=1
                return min(minimax(newAgent,depth,state.generateSuccessor(agent,newstate)) for newstate in state.getLegalActions(agent))

        _max = float("-inf")
        action = Directions.WEST
        for state in gameState.getLegalActions(0):
            numGet = minimax(1,0,gameState.generateSuccessor(0,state))
            if numGet > _max or _max == float("-inf"):
                _max = numGet
                action = state
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def max_value(agent,depth,state,a,b):
            v= float("-inf")
            for newstate in state.getLegalActions(agent):
                v=max(v,alpha_beta_prunning(1,depth,state.generateSuccessor(agent,newstate),a,b))
                if v > b:
                    return v
                a = max (a,v)
            return v

        def min_value(agent,depth,state,a,b):
            v = float("inf")
            newagent = agent + 1
            if newagent == state.getNumAgents():
                newagent = 0
            if newagent == 0:
                depth+=1
            for newstate in state.getLegalActions(agent):
                v = min(v,alpha_beta_prunning(newagent, depth, state.generateSuccessor(agent, newstate), a, b))
                if v < a:
                    return v
                b = min(b, v)
            return v

        def alpha_beta_prunning(agent,depth,state,a,b):
            if state.isWin() or state.isLose() or depth==self.depth :
                return self.evaluationFunction(state)
            if agent == 0:
                return max_value(agent, depth, state, a, b)
            else:
                return min_value(agent, depth, state, a, b)

        alpha = float("-inf")
        beta = float("inf")
        action = Directions.WEST
        _max = float("-inf")
        for _action in gameState.getLegalActions(0):
            newGet = alpha_beta_prunning(1,0,gameState.generateSuccessor(0,_action),alpha,beta)
            if newGet>_max:
                _max = newGet
                action = _action
            if newGet > beta :
                return action
            alpha = max(alpha,_max)
        return action

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

        def expectimax(agent,depth,state):
            if state.isWin() or state.isLose() or depth==self.depth:
                return self.evaluationFunction(state)
            if agent== 0:
                return max(expectimax(1,depth,state.generateSuccessor(0,_state)) for _state in state.getLegalActions(0))
            else:
                newAgent = agent + 1
                if newAgent == state.getNumAgents():
                    newAgent=0
                if newAgent==0:
                    depth+=1
                return sum(expectimax(newAgent,depth,state.generateSuccessor(agent,newstate)) for newstate in state.getLegalActions(agent))/float(len(state.getLegalActions(agent)))

        _max = float("-inf")
        action = Directions.WEST
        for state in gameState.getLegalActions(0):
            numGet = expectimax(1,0,gameState.generateSuccessor(0,state))
            if numGet > _max or _max == float("-inf"):
                _max = numGet
                action = state
        return action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    allFood = currentFood.asList()
    newScore = currentGameState.getScore()

    minDistance = -1
    for food in allFood:
        if minDistance > util.manhattanDistance(currentPos, food) or minDistance == -1:
            minDistance = util.manhattanDistance(currentPos, food)

    totalghostDistance = 1
    mayDie = 0
    for ghost in currentGameState.getGhostPositions():
        ghostDistance = util.manhattanDistance(currentPos, ghost)
        totalghostDistance += ghostDistance
        if ghostDistance <= 1:
            mayDie += 1

    remainFood = len(currentGameState.getCapsules())
    return newScore + (1 / float(minDistance)) - (1 / float(totalghostDistance)) - mayDie -remainFood
# Abbreviation
better = betterEvaluationFunction