from abc import ABC, abstractmethod
import numpy
import scipy.stats

class LrComputer(ABC):
    def __init__(self, G):
        self.G = G

        self.envs = list(self.G.nodes)
        self.num_envs = len(self.envs)
        self.timestep = 0
        self.timesteps = [[] for _ in range(self.num_envs)]
        self.returns = [[] for _ in range(self.num_envs)]
        self.lrs = numpy.zeros((self.num_envs))
    
    @abstractmethod
    def __call__(self, env_id, returnn):
        self.timestep += 1
        self.timesteps[env_id].append(self.timestep)
        self.returns[env_id].append(returnn)        

class TSLrComputer(LrComputer):
    def __init__(self, G, α):
        super().__init__(G)

        self.α = α

    @abstractmethod
    def compute_direct_lr(self, env_id):
        pass

    def __call__(self, env_id, returnn):
        super().__call__(env_id, returnn)
        lr = self.compute_direct_lr(env_id)
        if lr is not None:
            self.lrs[env_id] = self.α * lr + (1 - self.α) * self.lrs[env_id]
        return self.lrs

class OnlineLrComputer(TSLrComputer):
    def compute_direct_lr(self, env_id):
        returns = self.returns[env_id]
        if len(returns) >= 2:
            return returns[-1] - returns[-2]

class AbsOnlineLrComputer(OnlineLrComputer):
    def compute_direct_lr(self, env_id):
        lr = super().compute_direct_lr(env_id)
        if lr is not None:
            return abs(lr)

class WindowLrComputer(TSLrComputer):
    def __init__(self, G, α, N):
        super().__init__(G, α)

        self.N = N

    def compute_direct_lr(self, env_id):
        timesteps = self.timesteps[env_id][-self.N:]
        returns = self.returns[env_id][-self.N:]
        if len(timesteps) >= 2:
            return scipy.stats.linregress(timesteps, returns)[0]

class AbsWindowLrComputer(WindowLrComputer):
    def compute_direct_lr(self, env_id):
        lr = super().compute_direct_lr(env_id)
        if lr is not None:
            return abs(lr)

class AbsLinregLrComputer(LrComputer):
    def __init__(self, G, N):
        super().__init__(G)

        self.N = N

    def __call__(self, env_id, returnn):
        super().__call__(env_id, returnn)
        timesteps = self.timesteps[env_id][-self.N:]
        returns = self.returns[env_id][-self.N:]
        if len(timesteps) >= 2:
            self.lrs[env_id] = scipy.stats.linregress(timesteps, returns)[0]
        return self.lrs