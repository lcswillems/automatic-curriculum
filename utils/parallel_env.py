"""
Implements a class of parallel supervised environments for synchronous training. Heavily influenced by
https://github.com/lcswillems/torch-ac/blob/master/torch_ac/utils/penv.py
"""

from multiprocessing import Process, Pipe


def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "train_epoch":
            results = env.train_epoch(*data)
            conn.send(results)
        else:
            raise NotImplementedError


class ParallelEnv:
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    def train_epoch(self, model, encoder_optimizer, decoder_optimizer, criterion, epoch_length=10, batch_size=4096,
                    eval_everything=None, validate_using=None):
        args = [model, encoder_optimizer, decoder_optimizer, criterion, epoch_length, batch_size, eval_everything,
                validate_using]
        for local in self.locals:
            local.send(("train_epoch", args))
        results_0 = self.envs[0].train_epoch(*args)
        results = list(zip(*[results_0] + [local.recv() for local in self.locals]))
        # results is a list of 6 tuples, corresponding to the 6 outputs of AdditionEnv.train_epoch
        return results
