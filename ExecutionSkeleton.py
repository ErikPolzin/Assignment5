import argparse
from dataclasses import dataclass
import time
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from FourRooms import FourRooms

class FourRoomsAgent(object):
    """A RL Agent designed to find packages distributed over a map with boundaries."""

    @dataclass
    class State(object):
        """Characterize a particular agent state."""

        x: int
        y: int
        numPackagesRemaining: int

        def __hash__(self) -> int:
            """Hash for state lookup."""
            return hash((self.x, self.y, self.numPackagesRemaining))

    def __init__(self,
                 env: FourRooms,
                 learning_rate: Optional[float] = None,
                 discount_rate: Optional[float] = None):
        self.env = env
        self.learning_rate = learning_rate or DEFAULT_PARAMS[env.scenario][0]
        self.discount_rate = discount_rate or DEFAULT_PARAMS[env.scenario][1]
        self.V: Dict[FourRoomsAgent.State, Dict[int, float]] = {}
        for s in self.allPossibleStates():
            # Initialize random values for each state-action pair
            self.V[s] = {i: np.random.random() for i in range(4)}

    def allPossibleStates(self) -> Iterable[State]:
        """Iterate over every possible state that this agent can have.

        13 cells x 13 cells x 3 npackages = 507
        (and that times four actions, for the state-action pairs)
        """
        for y in range(13):
            for x in range(13):
                for n in range(self.env.getPackagesRemaining()):
                    yield FourRoomsAgent.State(x=x, y=y, numPackagesRemaining=n+1)

    def getBestAction(self) -> int:
        """Calculate the the best action in the current state."""
        bestValue = -float("inf")
        bestAction = 0  # Default to UP
        # Look at the value of each action in the current state
        for (a, v) in self.V[self.state()].items():
            if v > bestValue:
                bestAction = a
                bestValue = v
        return bestAction

    def takeAction(self, action: int) -> int:
        """Take an action in the current environment and return the reward."""
        result = self.env.takeAction(action)
        return self.reward(result[0], result[2])

    def updateV(self, state: State, action: int, r: int) -> None:
        """Update the value of taking a particular action from a certain state.

        Done according to the formula:
            q[s0, a] = (1 - alpha) * q[s0, a] + alpha * (reward + gamma * np.max(q[s1, :]))
        """
        s1 = self.state()
        if s1.numPackagesRemaining == 0:
            # If the number of packages remaining is 0 after the next move,
            # the result is terminal. Can safely ignore
            return
        v0 = self.V[state][action]
        v1_max = max(self.V[s1].values())
        a = self.learning_rate
        gamma = self.discount_rate
        self.V[state][action] = (1-a)*v0 + a*(r+gamma*v1_max)

    def reward(self, grid_cell: int, num_packages: int) -> int:
        """Calculate reward of taking a given action."""
        if grid_cell > 1:  # Cell is a package
            if self.env.scenario == "rgb":
                if grid_cell == 3-num_packages:
                    return 50  # Package is being collected in order
                return -1000  # Package collected out of order. BAD
            return 50  # Not in RGB mode, can be collected in any order
        # Deincentivise bumping into corners. Would be good, but currently
        # the framework doesn't report border cells (when trying to move into the
        # boundary it doesn't change the pos and returns the old cell value)
        elif grid_cell == FourRooms.BORDER:
            return -10
        return 0

    def state(self) -> State:
        """Return the agent's current state."""
        x, y = self.env.getPosition()
        n = self.env.getPackagesRemaining()
        return FourRoomsAgent.State(x=x, y=y, numPackagesRemaining=n)



    # Create FourRooms Object
    fourRoomsObj = FourRooms('simple')

    # This will try to draw a zero
    actSeq = [FourRooms.LEFT, FourRooms.LEFT, FourRooms.LEFT,
              FourRooms.UP, FourRooms.UP, FourRooms.UP,
              FourRooms.RIGHT, FourRooms.RIGHT, FourRooms.RIGHT,
              FourRooms.DOWN, FourRooms.DOWN, FourRooms.DOWN]

    aTypes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    gTypes = ['EMPTY', 'RED', 'GREEN', 'BLUE']

    print('Agent starts at: {0}'.format(fourRoomsObj.getPosition()))

    for act in actSeq:
        gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(act)

        print("Agent took {0} action and moved to {1} of type {2}".format (aTypes[act], newPos, gTypes[gridType]))

        if isTerminal:
            break

    # Don't forget to call newEpoch when you start a new simulation run

    # Show Path
    fourRoomsObj.showPath(-1)


if __name__ == "__main__":
    main()
