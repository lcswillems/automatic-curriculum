import argparse
import time

from envs import str_to_envs
import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Env-D4LuIuBu",
                    help="name of the environment to be run (default: Env-D4LuIuBu)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--deterministic", action="store_true", default=False,
                    help="action with highest probability is selected")
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment

env = str_to_envs(args.env, args.seed)[0]

# Define agent

agent = utils.Agent(args.model, env.observation_space, env.action_space, args.deterministic)

# Run the agent

def reset():
    obs = env.reset()
    print("Instr:", obs["mission"])
    return obs

obs = reset()

while True:
    time.sleep(0.1)
    renderer = env.render("human")

    action = agent.get_action(obs)
    obs, reward, done, _ = env.step(action)
    agent.analyze_feedback(reward, done)

    if done:
        obs = reset()

    if renderer.window is None:
        break