## Curriculums

### BlockedUnlockPickup curriculum

TODO: Images + graph

### KeyCorridor curriculum

TODO: Images + graph

### ObstructedMaze curriculum

TODO: Images + graph

## Training

```
python -m scripts.train_sl --curriculum Addition3  --model Addition3
```

```
python -m scripts.train_rl --curriculum BlockedUnlockPickup --model BlockedUnlockPickup
```

```
tensorboard --logdir=storage/models/BlockedUnlockPickup
```

## Evaluate

```
python -m scripts.evaluate_sl --gen Addition3 --model Addition3
```

```
python -m scripts.evaluate_rl --env MiniGrid-BlockedUnlockPickup-v0 --model BlockedUnlockPickup
```

## Visualize

```
python -m scripts.visualize_rl --env MiniGrid-BlockedUnlockPickup-v0 --model BlockedUnlockPickup
```
