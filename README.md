# Automatic curriculum

This repository contains:

- several curriculum algorithms (in `auto_curri`), i.e.:
  - the ones introduced in the ["Teacher-Student Curriculum Learning" paper](https://arxiv.org/abs/1707.00183)
  - the ones introduced in the "Mastering Rate based Curriculum Learning" paper
- several curriculums (in `curriculums`), i.e.:
  - 3 SL curriculums: `Addition3`, `Addition5`, `Addition9`
  - 3 RL curriculums: `BlockedUnlockPickup`, `KeyCorridor`, `ObstructedMaze`
- several scripts (in `scripts`) to easily:
  - train a model, using the curriculms and curriculum algorithms of the repository
  - evaluate it
  - visualize it (for RL only)

## Curriculums

Curriculums are represented by directed acyclic graphs on tasks. For every task are given estimations of the minimum and maximum possible performances on them. They are useful for the mastering rate based curriculum algorithms.

### Supervised Learning curriculums

This repository contains 3 versions of the Addition curriculum: `Addition3`, `Addition5` and `Addition9`.

The `Addition9` curriculum consists in learning how to sum two 9-digits numbers by first learning to sum two 1-digit number, then two 2-digits numbers, etc...

`Addition3` and `Addition5` curriculums are weaker versions of it, for respectively learning to sum two 3-digits numbers and two 5-digits numbers.

### Reinforcement Learning curriculums

This repository contains:
- `BlockedUnlockPickup`: [description and pictures](https://github.com/maximecb/gym-minigrid#blocked-unlock-pickup-environment).
- `KeyCorridor`: [description and pictures](https://github.com/maximecb/gym-minigrid#key-corridor-environment).
- `ObstructedMaze`: [description and pictures](https://github.com/maximecb/gym-minigrid#obstructed-maze-environment).

## Training scripts

Two scripts (for SL and RL settings) are available for training a model, either on a task or on a curriculum of tasks.

**For SL:**

A command for training on the `Addition4` data generator:

```
python -m scripts.train_sl --gen Addition4
```

A command for training on the `Addition3` curriculum with the default curriculum algorithm and setting model's name to `Addition3`:

```
python -m scripts.train_sl --curriculum Addition3  --model Addition3
```

A lot of arguments are available to select the good curriculum algorithm, learning rate, etc...

**For RL:**

A command for training on the MiniGrid `Unlock` environment:

```
python -m scripts.train_sl --env MiniGrid-Unlock-v0
```

A command for training on the `BlockedUnclockPickup` curriculum with the default curriculum algorithm and setting model's name to `BlockedUnlockPickup`:

```
python -m scripts.train_rl --curriculum BlockedUnlockPickup --model BlockedUnlockPickup
```

Again, a lot of arguments are available to choose the correct default curriculum algorithm, learning rate, etc...

**Visualization**

For visualizing the training progress of a particular model, execute the following command:

```
tensorboard --logdir=storage/models/BlockedUnlockPickup
```

## Evaluation scripts

Two scripts (for SL and RL settings) are available for evaluating a model on a task.

**For SL:**

A command for evaluating the `Addition3` model on the `Addition3` task on 100 examples:

```
python -m scripts.evaluate_sl --gen Addition3 --model Addition3
```

**For RL:**

A command for evaluating the `BlockedUnlockPickup` model on the `BlockedUnlockPickup` task during 100 episodes:

```
python -m scripts.evaluate_rl --env MiniGrid-BlockedUnlockPickup-v0 --model BlockedUnlockPickup
```

## Visualization scripts

A script for visualizing how the learner behaves is avaibable in the RL setting:

```
python -m scripts.visualize_rl --env MiniGrid-BlockedUnlockPickup-v0 --model BlockedUnlockPickup
```
