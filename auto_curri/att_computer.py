from abc import ABC, abstractmethod
import numpy
import networkx as nx


class AttentionComputer(ABC):
    """The abstraction of an attention computer.

    It computes an attention over tasks given the performance histories."""

    def __init__(self, perf_hists):
        self.perf_hists = perf_hists

        self.step = 0

    @abstractmethod
    def __call__(self, perfs):
        self.step += 1
        for task_idx, perf in perfs.items():
            self.perf_hists[task_idx].append(self.step, perf)


class LpAttentionComputer(AttentionComputer):
    """A learning progress based attention computer.

    Attention is computed with:
        a(c) = a_lp(c)
    where a_lp(c) is an estimate of the absolute learning progress on task c."""

    def __init__(self, perf_hists, estimate_lp):
        super().__init__(perf_hists)

        self.estimate_lp = estimate_lp

    def __call__(self, perfs):
        super().__call__(perfs)

        self.lps = self.estimate_lp()
        self.a_lps = numpy.absolute(self.lps)
        self.atts = self.a_lps

        return self.atts


class MrAttentionComputer(AttentionComputer):
    """A mastering rate based attention computer.

    It first associates a pre-attention pre_a(c) to each task i:
        pre_a(c) = Mast(Anc_c)^p * ((1-γ) na_lp(c) + γ (1 - Mast(c))) * (1 - Mast(Succ_c))
    where:
        - Mast(Anc_c) is the minimum mastering rate of ancestors of task c in graph G;
        - p is a power;
        - na_lp(c) := a_lp(c) / max_c a_lp(c) is the normalized absolute learning progress on c;
        - Mast(c) is the mastering rate of task c;
        - γ is the potential proportion;
        - Mast(Succ_c) is the mastering rate of successors of task c in graph G.

    Then, each task c gives recursively δ_pred of its pre-attention to its predecessors and
    gives not recursively δ_succ to its successors. This leads to the final attention a."""

    def __init__(self, perf_hists, estimate_lp, G, init_min_perfs, init_max_perfs,
                 K, power, pot_prop, δ_pred, δ_succ):
        super().__init__(perf_hists)

        self.min_perfs = numpy.array(init_min_perfs, dtype=numpy.float)
        self.max_perfs = numpy.array(init_max_perfs, dtype=numpy.float)
        self.K = K
        self.estimate_lp = estimate_lp
        self.G = G
        self.power = power
        self.pot_prop = pot_prop
        self.δ_pred = δ_pred
        self.δ_succ = δ_succ

        self.perfs = numpy.copy(self.min_perfs)

    def update_perfs(self):
        for i in range(len(self.perfs)):
            _, perfs = self.perf_hists[i][-self.K:]
            if len(perfs) > 0:
                mean_perf = numpy.mean(perfs[-self.K:])
                if len(perfs) >= self.K:
                    self.min_perfs[i] = min(self.min_perfs[i], mean_perf)
                    self.max_perfs[i] = max(self.max_perfs[i], mean_perf)
                self.perfs[i] = numpy.clip(mean_perf, self.min_perfs[i], self.max_perfs[i])

    def __call__(self, perfs):
        super().__call__(perfs)

        self.update_perfs()

        self.lps = self.estimate_lp()
        self.a_lps = numpy.absolute(self.lps)
        self.na_lps = self.a_lps / numpy.amax(self.a_lps) if numpy.amax(self.a_lps) != 0 else self.a_lps
        self.mrs = (self.perfs - self.min_perfs) / (self.max_perfs - self.min_perfs)
        self.pots = 1 - self.mrs
        self.anc_mrs = numpy.ones(len(self.perf_hists))
        for task_idx in self.G.nodes:
            anc_idxs = list(nx.ancestors(self.G, task_idx))
            if len(anc_idxs) > 0:
                self.anc_mrs[task_idx] = numpy.amin(self.mrs[anc_idxs])
        self.succ_mrs = numpy.zeros(len(self.perf_hists))
        for task_idx in self.G.nodes:
            succ_idxs = list(self.G.successors(task_idx))
            if len(succ_idxs) > 0:
                self.succ_mrs[task_idx] = numpy.amin(self.mrs[succ_idxs])
        self.learning_states = (1 - self.pot_prop) * self.na_lps + self.pot_prop * self.pots
        self.pre_atts = self.anc_mrs ** self.power * self.learning_states * (1 - self.succ_mrs)

        self.atts = numpy.copy(self.pre_atts)
        for task_idx in reversed(list(nx.topological_sort(self.G))):
            pred_idxs = list(self.G.predecessors(task_idx))
            att_to_preds = self.atts[task_idx] * self.δ_pred
            self.atts[task_idx] -= att_to_preds
            if len(pred_idxs) > 0:
                self.atts[pred_idxs] += att_to_preds / len(pred_idxs)
            succ_idxs = list(self.G.successors(task_idx))
            att_to_succs = self.atts[task_idx] * self.δ_succ
            self.atts[task_idx] -= att_to_succs
            if len(succ_idxs) > 0:
                self.atts[succ_idxs] += att_to_succs / len(succ_idxs)

        return self.atts
