import numpy as np
import torch

class EarlyStopping:
    def __init__(self, mode='min', patience=7, verbose=False, delta=0, path='weights/cifar10.pth', trace_func=print):
        self.mode = mode
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.val_score = np.Inf

    def save_checkpoint(self, score, model):
        if self.verbose:
            self.trace_func(f'Validtion score decrease: {self.val_score:.4f} -> {self.self.best_score:.4f}')
        torch.save(model.state_dict(), f=self.path)
        self.val_score = score

    def __call__(self, score, model):
        if self.mode == 'min':
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)

        elif score < self.best_score:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True
            else:
                self.trace_func(f'early_stopping: {self.counter}/{self.patience}')

        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0