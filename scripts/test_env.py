import argparse
import time

import envs
import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="SEnv_D4LuIuBu",
                    help="name of the environment to test (default: SEnv_D4LuIuBu)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment

env = envs.get_env(args.env, args.seed)

# Display interactive environment

def reset():
    obs = env.reset()
    print("Instr:", obs["mission"])

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

    _, _, done, _ = env.step(action)

    if done:
        reset()

reset()

while True:
    time.sleep(0.01)
    renderer = env.render('human')
    renderer.window.setKeyDownCb(keyDownCb)

    if renderer.window == None:
        break