import numpy as np

import joblib
from tqdm.auto import tqdm

class ProgressParallel(joblib.Parallel):
    def __init__(self, total=None, **kwds):
        self.total = total
        super().__init__(**kwds)
    def __call__(self, *args, **kwargs):
        with tqdm() as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self.total is None:
            self._pbar.total = self.n_dispatched_tasks
        else:
            self._pbar.total = self.total
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

def proper_round_float(val):
    if (float(val) % 1) >= 0.5:
        x = int(np.ceil(val))
    else:
        x = round(val)
    return x

def proper_round(val):
    if isinstance(val, np.ndarray):
        return np.array([proper_round_float(v) for v in val],dtype=int)
    else:
        return proper_round_float(val)
