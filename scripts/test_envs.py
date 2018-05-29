#!/usr/bin/env python3

import time
import argparse

from env import Env

parser = argparse.ArgumentParser()
parser.add_argument("--doors", type=int, default=4)
parser.add_argument("--locked", type=float, default=0.5)
parser.add_argument("--inbox", type=float, default=0.5)
parser.add_argument("--blocked", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

env = Env(args.doors, args.locked, args.inbox, args.blocked, args.seed)

def reset():
    obs = env.reset()
    print("instr:", obs["mission"])

def keyDownCb(keyName):
    if keyName == 'BACKSPACE':
        reset()
        return

    if keyName == 'LEFT':
        action = env.actions.left
    elif keyName == 'RIGHT':
        action = env.actions.right
    elif keyName == 'UP':
        action = env.actions.forward
    elif keyName == 'SPACE':
        action = env.actions.toggle
    elif keyName == 'PAGE_UP':
        action = env.actions.pickup
    elif keyName == 'PAGE_DOWN':
        action = env.actions.drop
    else:
        return

    _, reward, done, _ = env.step(action)

    if done == True:
        print("reward:", reward)
        reset()

renderer = env.render('human')
renderer.window.setKeyDownCb(keyDownCb)
reset()

while True:
    env.render('human')
    time.sleep(0.01)

    if renderer.window == None:
        break