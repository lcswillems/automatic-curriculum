## Curriculums

### BlockedUnlockPickup curriculum

TODO: Images + graph

### KeyCorridor curriculum

TODO: Images + graph

### ObstructedMaze curriculum

TODO: Images + graph

## Training

```
python -m scripts.train --curriculum BlockedUnlockPickup --model BlockedUnlockPickup --save-interval 10 --epochs 1000
```

```
tensorboard --logdir=storage/models/BlockedUnlockPickup
```

## Evaluate

```
python -m scripts.evaluate --env MiniGrid-BlockedUnlockPickup-v0 --model BlockedUnlockPickup
```

## Visualize

```
python -m scripts.visualize --env MiniGrid-BlockedUnlockPickup-v0 --model BlockedUnlockPickup
```