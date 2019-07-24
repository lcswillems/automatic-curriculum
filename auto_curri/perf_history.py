from abc import ABC, abstractmethod
import numpy


class PerformanceHistory:
    """The performance history.

    It tracks the performance achieved on a task over time."""

    def __init__(self):
        self.steps = []
        self.perfs = []

    def append(self, step, perf):
        self.steps.append(step)
        self.perfs.append(perf)

    def __getitem__(self, index):
        return self.steps[index], self.perfs[index]
