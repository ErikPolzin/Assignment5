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


def run(scenario: str,
        stochastic: bool = False,
        epochs: int = 20,
        learning_rate: Optional[float] = None,
        discount_rate: Optional[float] = None,
        feedback: bool = False,
        max_actions: int = 10000) -> Tuple[FourRooms, float]:
    """Run a given scenario and return the average moves before picking up all packages."""

    # Create FourRooms Object
    fourRoomsObj = FourRooms(scenario, stochastic=stochastic)
    agent = FourRoomsAgent(fourRoomsObj, learning_rate, discount_rate)
    fitness = 0.0

    for epoch in range(epochs):
        i = 0
        fourRoomsObj.newEpoch()
        N = fourRoomsObj.getPackagesRemaining()
        # Cap the number of moves at max_actions, in case of infinite loops
        while not fourRoomsObj.isTerminal() and i < max_actions:
            prevState = agent.state()
            nextAction = agent.getBestAction()
            reward = agent.takeAction(nextAction)
            agent.updateV(prevState, nextAction, reward)
            i += 1
        n = N - fourRoomsObj.getPackagesRemaining()
        fitness += 0.2*(1-i/max_actions)+0.8*(n/N)*100
        if feedback:
            print(f"Found {n} packages in {i} moves")
    return fourRoomsObj, fitness/epochs


def main():
    parser = argparse.ArgumentParser(
        prog='ExecutionSkeleton',
        description='Trains the FourRooms Agent in a given scenario')
    parser.add_argument('scenario')
    parser.add_argument('-stochastic', action='store_true')
    parser.add_argument('-learning-rate', default=None, type=float)
    parser.add_argument('-discount-rate', default=None, type=float)
    parser.add_argument('-epochs', default=20, type=int)
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-save', default=None, type=str)

    args = parser.parse_args()
    kwargs = {
        "stochastic": args.stochastic,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "discount_rate": args.discount_rate
    }
    if args.test:
        test(args.scenario, **kwargs)
    else:
        startTime = time.time()
        fourRoomsObj, fitness = run(args.scenario, feedback=True, **kwargs)
        elapsedTime = time.time() - startTime
        print(f"Ran with average fitness: {fitness:.0f}%")
        print(f"Execution time: {elapsedTime*1000:.0f}ms")
        fourRoomsObj.showPath(-1, args.save)  # Show Path


if __name__ == "__main__":
    main()
