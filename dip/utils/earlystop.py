import math

class EarlyStopper:
    def __init__(self, mode='max', patience=200, min_delta=0.0):
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best = -math.inf if mode=='max' else math.inf
        self.num_bad = 0
        self.best_state = None
        self.best_iter = 0

    def step(self, value, state=None, iteration=0):
        improved = (value > self.best + self.min_delta) if self.mode=='max' else (value < self.best - self.min_delta)
        if improved:
            self.best = value
            self.num_bad = 0
            self.best_state = state
            self.best_iter = iteration
        else:
            self.num_bad += 1
        return self.num_bad > self.patience
