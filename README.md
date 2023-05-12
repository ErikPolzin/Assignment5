# Package collection via reinforcement learning

Apply reinforcement learning to an agent in a 13x13 four-room bounded environment.
Runs under the following scenarios:
- 1: Single package
- 2: Multiple packages
- 3: Multiple packages with ordered collection

## Build & launch instructions

Run `make` inside the extracted folder. Pip will install its dependencies inside a virtual environment


## Command-line interface

```bash
usage: ExecutionSkeleton [-h] [-stochastic] [-learning-rate LEARNING_RATE] [-discount-rate DISCOUNT_RATE]
                         [-epochs EPOCHS] [-test] [-save SAVE]
                         scenario

Trains the FourRooms Agent in a given scenario

positional arguments:
  scenario

options:
  -h, --help            show this help message and exit
  -stochastic
  -learning-rate LEARNING_RATE
  -discount-rate DISCOUNT_RATE
  -epochs EPOCHS
  -test
  -save SAVE
```

## Files

- `FourRooms.py`: Environment framework
- `ExecutionSkeleton.py`: Agent definition and runners
- `Scenario1.py`: Run the `simple` scenario
- `Scenario2.py`: Run the `multi` scenario
- `Scenario3.py`: Run the `rgb` scenario