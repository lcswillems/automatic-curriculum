from .a2d_converter import *
from .att_computer import *
from .dist_computer import *
from .lp_estimator import *
from .perf_history import *
from .polyenv import *


def make_dist_computer(num_tasks, lpe="Linreg", lpe_alpha=0.1, lpe_K=10,
                       acp="MR", acp_MR_G=None, acp_MR_init_min_perfs=None,
                       acp_MR_init_max_perfs=None, acp_MR_K=10, acp_MR_power=6,
                       acp_MR_pot_prop=0.5, acp_MR_att_pred=0.2, acp_MR_att_succ=0.05,
                       a2d="Prop", a2d_eps=0.1, a2d_tau=4e-4):
    # Instantiate the performance histories
    perf_hists = [PerformanceHistory() for _ in range(num_tasks)]

    # Instantiate the good learning progress estimator
    if lpe == "Online":
        estimate_lp = OnlineLpEstimator(perf_hists, lpe_alpha)
    elif lpe == "Naive":
        estimate_lp = NaiveLpEstimator(perf_hists, lpe_alpha, lpe_K)
    elif lpe == "Window":
        estimate_lp = WindowLpEstimator(perf_hists, lpe_alpha, lpe_K)
    elif lpe == "Linreg":
        estimate_lp = LinregLpEstimator(perf_hists, lpe_K)
    elif lpe == "Sampling":
        estimate_lp = SamplingLpEstimator(perf_hists, lpe_K)

    # Instantiate the good attention computer
    if acp == "LP":
        compute_att = LpAttentionComputer(perf_hists, estimate_lp)
    elif acp == "MR":
        assert None not in [acp_MR_G, acp_MR_init_min_perfs, acp_MR_init_max_perfs]
        compute_att = MrAttentionComputer(perf_hists, estimate_lp, acp_MR_G, acp_MR_init_min_perfs,
                                          acp_MR_init_max_perfs, acp_MR_K, acp_MR_power, acp_MR_pot_prop,
                                          acp_MR_att_pred, acp_MR_att_succ)

    # Instantiate the good distribution converter
    if a2d == "Amax":
        convert_a2d = AmaxA2DConverter()
    elif a2d == "GreedyAmax":
        convert_a2d = GreedyAmaxA2DConverter(a2d_eps)
    elif a2d == "Prop":
        convert_a2d = PropA2DConverter()
    elif a2d == "GreedyProp":
        convert_a2d = GreedyPropA2DConverter(a2d_eps)
    elif a2d == "Boltzmann":
        convert_a2d = BoltzmannA2DConverter(a2d_tau)

    return DistributionComputer(compute_att, convert_a2d)
