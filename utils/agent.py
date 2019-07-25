import torch

import utils
from model import ACModel


class RLAgent:
    """A RL agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir, device=None, argmax=False, num_envs=1):
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space)
        self.acmodel = ACModel(obs_space, action_space)
        self.device = device
        self.argmax = argmax
        self.num_envs = num_envs

        self.acmodel.load_state_dict(utils.get_model_state(model_dir))
        self.acmodel.to(self.device)
        self.acmodel.eval()

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=self.device)

        with torch.no_grad():
            dist, _ = self.acmodel(preprocessed_obss)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        pass

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
