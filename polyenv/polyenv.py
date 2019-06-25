import multiprocessing as mp
import numpy


def recv_conns(conns):
    """Receives the data coming from all the connections."""

    data = []
    wconns = mp.connection.wait(conns, timeout=.0)
    while len(wconns) > 0:
        for r in wconns:
            data.append(r.recv())
        wconns = mp.connection.wait(conns, timeout=.0)
    return data


class PolyEnvHead:
    """The head of polymorph environments.

    It communicates with polymorph environments through pipes: it first
    receives (env_id, return) tuples from them, updates the distribution
    over their environments (with a DistComputer object) after some time
    and sends them this distribution.

    This class centralizes the distribution of the polymorph environments."""

    def __init__(self, num_penvs, num_envs, compute_dist=None):
        self.num_penvs = num_penvs
        self.num_envs = num_envs
        self.compute_dist = compute_dist

        self._init_connections()
        self._reset_returns()
        self.dist = numpy.ones(self.num_envs)/self.num_envs
        self.update_dist()

    def _init_connections(self):
        self.locals, self.remotes = zip(*[mp.Pipe() for _ in range(self.num_penvs)])

    def _reset_returns(self):
        self.returns = {env_id: [] for env_id in range(self.num_envs)}

    def _recv_returns(self):
        data = recv_conns(self.locals)
        for env_id, returnn in data:
            self.returns[env_id].append(returnn)

    def _synthesize_returns(self):
        self.synthesized_returns = {}
        for env_id, returnn in self.returns.items():
            if len(returnn) > 0:
                self.synthesized_returns[env_id] = numpy.mean(returnn)

    def _send_dist(self):
        for local in self.locals:
            local.send(self.dist)

    def update_dist(self):
        self._recv_returns()
        self._synthesize_returns()
        self._reset_returns()

        if self.compute_dist is not None:
            self.dist = self.compute_dist(self.synthesized_returns)

        self._send_dist()


class PolyEnv:
    """A polymorph environment.

    It simulates different environments: it receives a distribution
    from its head, samples an environment from it, simulates it and
    then sends a (env_id, return) tuple to its head."""

    def __init__(self, envs, head_conn, seed=None):
        self.envs = envs
        self.head_conn = head_conn
        self.rng = numpy.random.RandomState(seed)

        self.num_envs = len(envs)
        self.returnn = None
        self.reset()

    def __getattr__(self, key):
        return getattr(self.env, key)

    def _recv_dist(self):
        data = recv_conns([self.head_conn])
        if len(data) > 0:
            self.dist = data[-1]

    def _select_env(self):
        self._recv_dist()
        self.env_id = self.rng.choice(range(self.num_envs), p=self.dist)
        self.env = self.envs[self.env_id]

    def _send_return(self):
        if self.returnn is not None:
            self.head_conn.send((self.env_id, self.returnn))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.returnn += reward
        return obs, reward, done, info

    def reset(self):
        self._send_return()
        self.returnn = 0
        self._select_env()
        return self.env.reset()

    def render(self, mode="human"):
        return self.env.render(mode)


class PolySupervisedEnv:
    """Standalone implementation of polymorph environment without any multiprocessing whatsoever"""
    def __init__(self, envs, seed=None, compute_dist=None):
        self.envs = envs
        self.rng = numpy.random.RandomState(seed)
        self.compute_dist = compute_dist

        self.use_batch = not isinstance(envs, list)
        self.num_envs = (envs.number_of_digits[1] - envs.number_of_digits[0] + 1) if self.use_batch else len(envs)
        # self.returns = {}
        self.dist = numpy.ones(self.num_envs) / self.num_envs

        self._select_env()

    def __getattr__(self, key):
        return getattr(self.env, key)

    def update_dist(self):
        if self.compute_dist is not None:
            self.dist = self.compute_dist(self.synthesized_returns)

    def _select_env(self):
        if self.use_batch:
            self.env = self.envs
            self.env.probs = self.dist
            self.number_of_digits = -1
        else:
            self.env_id = self.rng.choice(range(self.num_envs), p=self.dist)
            self.env = self.envs[self.env_id]
            self.number_of_digits = self.env.number_of_digits

    def train_epoch(self, model, encoder_optimizer, decoder_optimizer, criterion, epoch_length=10, batch_size=4096,
                    eval_everything=None, validate_using=None):
        results = self.env.train_epoch(model, encoder_optimizer, decoder_optimizer, criterion, epoch_length,
                                       batch_size, eval_everything, validate_using)
        _, _, _, _, per_number_ac_test, test_results = results
        # self.returnn = per_number_ac_test
        self.test_results = test_results
        self.reset()
        return results

    def reset(self):
        # self.synthesized_returns = {self.env_id: self.returnn}
        self.synthesized_returns = {index: self.test_results[key][1] for index, key in enumerate(self.test_results)}
        self.update_dist()
        self._select_env()

