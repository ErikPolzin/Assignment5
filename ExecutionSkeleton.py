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

    def __init__(self, env: FourRooms, learning_rate: float, discount_rate: float):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
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


def test(scenario: str, stochastic: bool, epochs: int) -> None:
    """Test a range of parameters (LR, DR) to find the best combination."""
    best = -float("inf")
    bestparams = (-1, -1)
    # Test in the same room
    room = FourRooms(scenario, stochastic=stochastic)
    print(f"Testing parameters with {epochs} epochs")
    for lr in np.arange(0.05, 0.95, 0.05):
        for dr in np.arange(0.05, 0.95, 0.05):
            fitness = run(scenario,
                          stochastic=stochastic,
                          epochs=epochs,
                          learning_rate=lr,
                          discount_rate=dr,
                          room=room)[1]
            if fitness > best:
                print(f"LR: {lr:.3f}, DR: {dr:.3f} ({fitness:.0f}%)")
                best = fitness
                bestparams = (lr, dr)
            room.newEpoch()
    print(f"Best LR: {bestparams[0]:.3f}, DR: {bestparams[1]:.3f}")


def run(scenario: str,
        stochastic: bool,
        epochs: int,
        learning_rate: float,
        discount_rate: float,
        feedback: bool = False,
        room: Optional[FourRooms] = None,
        max_loops: int = 10000) -> Tuple[FourRooms, float]:
    """Run a given scenario and return the average moves before picking up all packages."""
    if room is None:
        # Create FourRooms Object
        room = FourRooms(scenario, stochastic=stochastic)
    agent = FourRoomsAgent(room, learning_rate, discount_rate)
    N = room.getPackagesRemaining()
    # The maximum number of actions before collecting a package in the opposing corner
    max_actions = 13*2*N
    avg_score = 0.0
    feedback_mod = epochs // 10

    if feedback:
        print("Running with learning_rate={:.3f}, discount_rate={:.3f}".format(
            agent.learning_rate, agent.discount_rate))

    for epoch in range(epochs):
        i = 0
        room.newEpoch()
        # Cap the number of moves at max_loops, in case of infinite loops
        while not room.isTerminal() and i < max_loops:
            prevState = agent.state()
            nextAction = agent.getBestAction()
            reward = agent.takeAction(nextAction)
            agent.updateV(prevState, nextAction, reward)
            i += 1
        n = N - room.getPackagesRemaining()
        pathScore = min(1, max(0, (1-i/max_actions)))
        packageScore = n/N
        score = (0.5*pathScore+0.5*packageScore)*100
        if feedback and (epoch % feedback_mod == 0):
            print(f"{epoch:<03}: Found {n} packages in {i} moves ({score:.0f}% score)")
        avg_score += score / epochs
    # Return room and score. Here efficient paths and package scores are weighted evenly
    return room, avg_score


def main():
    parser = argparse.ArgumentParser(
        prog='ExecutionSkeleton',
        description='Trains the FourRooms Agent in a given scenario')
    parser.add_argument('scenario')
    parser.add_argument('-stochastic', action='store_true')
    parser.add_argument('-learning-rate', default=0.6, type=float)
    parser.add_argument('-discount-rate', default=0.3, type=float)
    parser.add_argument('-epochs', default=300, type=int)
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-save', default=None, type=str)

    args = parser.parse_args()
    if args.test:
        test(args.scenario, args.stochastic, args.epochs)
    else:
        startTime = time.time()
        fourRoomsObj, fitness = run(
            args.scenario, args.stochastic, args.epochs,
            args.learning_rate, args.discount_rate, feedback=True)
        elapsedTime = time.time() - startTime
        print(f"Ran with average fitness: {fitness:.0f}%")
        print(f"Execution time: {elapsedTime*1000:.0f}ms")
        fourRoomsObj.showPath(-1, args.save)  # Show Path


if __name__ == "__main__":
    main()
