import random
import logging
import time

from pacai.util import util
from pacai.core.directions import Directions
from pacai.agents.capture.reflex import ReflexCaptureAgent


def createTeam(firstIndex, secondIndex, isRed, first = '', second=''):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = Kicker(firstIndex)
    secondAgent = Goalie(secondIndex)

    return [
        firstAgent,
        secondAgent,
    ]

class Goalie(ReflexCaptureAgent):
    """
    A reflex agent that tries to keep its side Pacman-free.
    This is to give you an idea of what a defensive agent could be like.
    It is not the best or only way to make such an agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getFeatures(self, gameState, action):
        features = {}

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # calculate distance from our teammate, if we need it
        players = [0, 1, 2, 3]
        partPos = [p for p in players if self.index != p
                and p not in self.getOpponents(successor)]
        pState = successor.getAgentState(partPos[0])
        partPos = pState.getPosition()
        partnerDistance = self.getMazeDistance(myPos, partPos)

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        # go after any pacman we see
        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            # print(invaders[0].getPosition())
            minDist = min(dists)
            if min(dists) > 4:
                minDist = minDist + 100
            features['invaderDistance'] = min(dists) + 100
        else:
            # features['partnerDistance'] = partnerDistance
            # dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            # minDist = min(dists)
            endangeredPellets = {}
            distancesFromGhosts = []
            for pellet in self.getFoodYouAreDefending(successor).asList():
                # print(pellet)
                minDistFromEnemies = 0
                for enemy in enemies:
                    # print("enemy from pellet", self.getMazeDistance(pellet, enemy.getPosition()))
                    if minDistFromEnemies < self.getMazeDistance(pellet, enemy.getPosition()):
                        minDistFromEnemies = self.getMazeDistance(pellet, enemy.getPosition())
                distancesFromGhosts.append(minDistFromEnemies)
                endangeredPellets[minDistFromEnemies] = pellet
                # print("opp", enemy.getPosition(), "is", minDistFromEnemies, "spots away from pellet", endangeredPellets[minDistFromEnemies])

            mostDangerPellet = endangeredPellets[min(distancesFromGhosts)]
            # print("IN DANGER", mostDangerPellet, "is", min(distancesFromGhosts), "spaces away from a ghost")
            features['endangeredPelletDistance'] = self.getMazeDistance(myPos, mostDangerPellet)

        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10000,
            'stop': -100,
            'reverse': -2,
            'endangeredPelletDistance': -10
        }

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest return from `ReflexCaptureAgent.evaluate`.
        """

        actions = gameState.getLegalActions(self.index)

        start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        logging.debug('evaluate() time for agent %d: %.4f' % (self.index, time.time() - start))

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        if (pos != util.nearestPoint(pos)):
            # Only half a grid position was covered.
            return successor.generateSuccessor(self.index, action)
        else:
            return successor


class Kicker(ReflexCaptureAgent):
    """
    A reflex agent that seeks food.
    This agent will give you an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getFeatures(self, gameState, action):
        features = {}
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        return features

    def getWeights(self, gameState, action):
        return {
            'successorScore': 100,
            'distanceToFood': -1
        }

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest return from `ReflexCaptureAgent.evaluate`.
        """

        # actions = gameState.getLegalActions(self.index)

        start = time.time()
        # lets do minimax here, instead of just picking the higest value

        actions = gameState.getLegalActions(self.index)
        ways = []
        for action in actions:
            bestAction = self.value(gameState, action, self.index + 1,
                                    self.index, -999999999, 999999999)
            # print("best action ", bestAction)
            ways.append(bestAction)

        bestWays = []
        bestVal = -9999999999999999
        for move in ways:
            if bestVal < move[0]:
                bestVal = move[0]
                bestWays.append(move)

        # print("bestWays:", bestWays)
        logging.debug('evaluate() time for agent %d: %.4f' % (self.index, time.time() - start))

        return random.choice(bestWays)[1]

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        if (pos != util.nearestPoint(pos)):
            # Only half a grid position was covered.
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    # prune via alpha and beta values to return quicker
    def minValue(self, state, move, depp, aid, alpha, beta):
        legalMoves = state.getLegalActions(aid)
        if 'Stop' in legalMoves:
            legalMoves.remove("Stop")
        # print(legalMoves)

        minUtil = 999999969
        minAction = ""
        for move in legalMoves:
            # print(move)
            succState = state.generateSuccessor(aid, move)
            succUtil, succAction = self.value(succState, move, depp, aid + 1, alpha, beta)

            if minAction == "" or succUtil <= minUtil:
                minAction = move
                minUtil = succUtil

            if minUtil <= alpha:
                # print(minUtil, alpha)
                return (minUtil, minAction)

            beta = min(beta, minUtil)

            # print("MIN", aid, "Move:", move, "successor:", (succAction, succUtil))

        return (minUtil, minAction)

    def maxValue(self, state, move, depp, aid, alpha, beta):
        legalMoves = state.getLegalActions(aid)
        if 'Stop' in legalMoves:
            legalMoves.remove("Stop")
        # print(legalMoves)

        maxUtil = -999999999
        maxAction = ""
        for move in legalMoves:
            # print(move)
            succState = state.generateSuccessor(aid, move)
            succUtil, succAction = self.value(succState, move, depp, aid + 1, alpha, beta)

            if maxAction == "" or succUtil > maxUtil:
                maxAction = move
                maxUtil = succUtil

            if maxUtil >= beta:
                # print(maxUtil, beta)
                return (maxUtil, maxAction)

            alpha = max(alpha, maxUtil)

            # print("MAX", aid, "Move:", move, "successor:", (succAction, succUtil))

        return (maxUtil, maxAction)

    def value(self, state, move, depp, aid, alpha, beta):
        if aid == state.getNumAgents():
            aid = self.index
            depp += 1

        if state.isWin() or state.isLose():
            return 0, "term"

        if depp > self.getTreeDepth():
            legalMoves = state.getLegalActions(aid)
            # if 'Stop' in legalMoves: legalMoves.remove("Stop")
            # print(aid, legalMoves)
            ways = [self.evaluate(state, m) for m in legalMoves]
            return max(ways), "term"

        # print("agent", aid)
        if aid in self.getTeam(state):
            # print("PACAI!")
            return self.maxValue(state, move, depp, aid, alpha, beta)
        if aid in self.getOpponents(state):
            # print("GHOST!")
            return self.minValue(state, move, depp, aid, alpha, beta)

    def getTreeDepth(self):
        return 1 + self.index
