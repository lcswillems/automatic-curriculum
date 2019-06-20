import numpy
import torch
import torch_ac


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
    return 20 * reward


def synthesize_supervised_results(results):
    """
    :param results: list of 6 tuples. Each tuple contain as many floats as there are processes. The last tuple contain
    dicts with same keys. Useful for multiprocessing only.
    :return: means
    """
    # TODO: improve this, group by num_digit or something
    modified_results = []
    for tup in results:
        if type(tup[0]) == float:
            modified_results.append(numpy.mean(tup))
        elif isinstance(tup[0], dict):
            keys = tup[0].keys()
            new_dict = {}
            for key in keys:
                new_dict[key] = numpy.mean((dic[key] for dic in tup))
            modified_results.append(new_dict)
        else:
            raise NotImplementedError

    return modified_results