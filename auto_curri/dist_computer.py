class DistributionComputer:
    """A distribution computer.

    It first computes an attention over tasks given the performance histories
    and then converts it into a distribution over tasks."""

    def __init__(self, compute_att, convert_a2d):
        self.compute_att = compute_att
        self.convert_a2d = convert_a2d

    def __call__(self, perfs):
        return self.convert_a2d(self.compute_att(perfs))
