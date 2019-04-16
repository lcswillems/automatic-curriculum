import os
import json
import numpy
import re
import torch
import torch_ac
import gym

import utils

def get_obss_preprocessor(env_id, obs_space, model_dir):
    obs_space = {"image": obs_space.spaces['image'].shape}

    def preprocess_obss(obss, device=None):
        return torch_ac.DictList({
            "image": preprocess_images([obs["image"] for obs in obss], device=device)
        })

    return obs_space, preprocess_obss

def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = numpy.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)

def reshape_reward(obs, action, reward, done):
    return 20*reward